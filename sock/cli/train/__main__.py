import argparse
import logging
import math
import random
from argparse import ArgumentParser, ArgumentTypeError, FileType

import ignite
import torch
from ignite.engine import Engine, Events, State, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, Timer
from ignite.metrics import BinaryAccuracy, Loss, Precision, Recall
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from sock.model.data import WordEmbeddings, tokenize
from sock.model.data.batching import sentence_label_pad, sentence_pad
from sock.model.dataset import (CresciTweetDataset, Five38TweetDataset, LabelDataset, NbcTweetDataset,
                                SingleLabelDataset, TweetTensorDataset)
from sock.model.nn import ContextualLSTM
from sock.model.serial import load, save
from sock.utils import BOT, NOT_BOT, Metrics, Splits, expand_binary_class, split_integers, to_singleton_row


def positive_int(arg: str) -> int:
    i = int(arg)

    if i <= 0:
        raise ArgumentTypeError(f"{i} is not a positive integer")

    return i


def positive_finite_float(arg: str) -> float:
    f = float(arg)

    if f <= 0 or math.isnan(f) or math.isinf(f):
        raise ArgumentTypeError(f"{f} is not a positive and finite number")

    return f


def nonzero_finite_float(arg: str) -> float:
    f = float(arg)

    if math.isnan(f) or math.isinf(f):
        raise ArgumentTypeError(f"{f} is not a finite nonzero number")

    return f


def nonzero_fraction(arg: str) -> float:
    f = float(arg)

    if f <= 0.0 or f >= 1.0:
        raise ArgumentTypeError(f"{f} is not between 0 and 1 (exclusive)")

    return f


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Train a model"
    )

    data_args = parser.add_argument_group("Data")
    data_args.add_argument(
        "--glove",
        help="The word vector embeddings to use",
        metavar="path",
        type=FileType('r', encoding="utf8"),
        required=True
    )

    data_args.add_argument(
        "--bots",
        help="One or more files containing tweets known to be from bots",
        metavar="path",
        type=FileType('r', encoding="utf8"),
        nargs="+",
        required=True
    )

    data_args.add_argument(
        "--humans",
        help="One or more files containing tweets known to be from humans",
        metavar="path",
        type=FileType('r', encoding="utf8"),
        nargs="+",
        required=True
    )

    data_args.add_argument(
        "--max-tweets",
        help="The maximum number of the given tweets to use in training the model.  Default: all tweets.",
        metavar="max",
        type=positive_int
    )

    data_args.add_argument(
        "--output",
        help="Location to save the trained model",
        metavar="out",
        type=FileType("wb"),
        required=True
    )

    optimizer_hyperparams = parser.add_argument_group("Optimizer Hyperparameters")

    optimizer_hyperparams.add_argument(
        "--lr",
        help="Learning rate (default: %(default)s)",
        type=positive_finite_float,
        default=1e-3,
        metavar="lr"
    )

    optimizer_hyperparams.add_argument(
        "--eps",
        help="Term added to the denominator to improve numerical stability (default: %(default)s)",
        type=positive_finite_float,
        default=1e-8,
        metavar="e"
    )

    optimizer_hyperparams.add_argument(
        "--beta0",
        help="First coefficient used for computing running averages of gradient and its square (default: %(default)s)",
        type=positive_finite_float,
        default=0.9,
        metavar="b0"
    )

    optimizer_hyperparams.add_argument(
        "--beta1",
        help="Second coefficient used for computing running averages of gradient and its square (default: %(default)s)",
        type=positive_finite_float,
        default=0.999,
        metavar="b1"
    )

    optimizer_hyperparams.add_argument(
        "--weight-decay",
        help="Weight decay (L2 penalty) (default: %(default)s)",
        type=nonzero_finite_float,
        default=0.0,
        metavar="wd"
    )

    optimizer_hyperparams.add_argument(
        "--amsgrad",
        help="Whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond (default: %(default)s)",
        action="store_true"
    )

    lr_hyperparams = parser.add_argument_group("LR Scheduler Hyperparameters")
    lr_hyperparams.add_argument(
        "--lr-patience",
        help="If no improvement after this many epochs, reduce the learning rate (default: %(default)s)",
        type=positive_int,
        default=3,
        metavar="patience"
    )

    training_hyperparams = parser.add_argument_group("Training Hyperparameters")
    training_hyperparams.add_argument(
        "--max-epochs",
        help="The maximum number of passes to make over the input data (default: %(default)s)",
        type=positive_int,
        default=50,
        metavar="epochs"
    )

    training_hyperparams.add_argument(
        "--trainer-patience",
        help="If no improvement after this many epochs, end the training (default: %(default)s)",
        type=positive_int,
        default=10,
        metavar="patience"
    )

    training_hyperparams.add_argument(
        "--batch-size",
        help="The number of tweets to process at once (default: %(default)s)",
        metavar="size",
        type=positive_int,
        default=500
    )

    training_hyperparams.add_argument(
        "--train-split",
        help="Fraction of input data set aside for training the model (default: %(default)s)",
        type=nonzero_fraction,
        default=0.5
    )

    training_hyperparams.add_argument(
        "--valid-split",
        help="Fraction of input data set aside for tuning hyperparameters (default: %(default)s)",
        type=nonzero_fraction,
        default=0.2
    )

    training_hyperparams.add_argument(
        "--test-split",
        help="Fraction of input data set aside for evaluating model performance (default: %(default)s)",
        type=nonzero_fraction,
        default=0.3
    )

    return parser


def validate_args(args):
    if args.beta0 >= args.beta1:
        raise ArgumentTypeError(f"{args.beta0} is not less than {args.beta1}")

    if args.train_split + args.valid_split + args.test_split != 1.0:
        raise ArgumentTypeError(f"{args.train_split}, {args.valid_split}, and {args.test_split} do not add to 1")


def load_tweets(file, embeddings: WordEmbeddings) -> Dataset:

    try:
        logging.debug("Loading %s as a Cresci-format dataset", file.name)
        tweets = CresciTweetDataset(file.name)
        logging.info("Loaded %s as a Cresci-format dataset (len=%d)", file.name, len(tweets))
        return TweetTensorDataset(tweets, tokenize, embeddings)
    except Exception as e:
        logging.debug("Couldn't load %s as a Cresci-format dataset: %s", file.name, e)

    try:
        logging.debug("Loading %s as a NBC-format dataset", file.name)
        tweets = NbcTweetDataset(file.name)
        logging.info("Loaded %s as a NBC-format dataset (len=%d)", file.name, len(tweets))
        return TweetTensorDataset(tweets, tokenize, embeddings)
    except Exception as e:
        logging.debug("Couldn't load %s as a NBC-format dataset: %s", file.name, e)

    try:
        logging.debug("Loading %s as a 538-format dataset", file.name)
        tweets = Five38TweetDataset(file.name)
        logging.info("Loaded %s as a 538-format dataset (len=%d)", file.name, len(tweets))
        return TweetTensorDataset(tweets, tokenize, embeddings)
    except Exception as e:
        logging.debug("Couldn't load %s as a 538-format dataset: %s", file.name, e)

    logging.error("Could not load %s as a tweet dataset!", file.name)
    raise ValueError(f"Could not load {file.name} as a tweet dataset")


def load_glove(args) -> WordEmbeddings:

    logging.info("Loading GloVe embeddings from %s", args.glove.name)
    embeddings = WordEmbeddings(args.glove, device="cuda")

    logging.info(
        "Loaded GloVe embeddings from %s (dim=%d, device=%s, len=%d)",
        args.glove.name,
        embeddings.dim,
        embeddings.device,
        len(embeddings)
    )

    return embeddings


def create_model(args, glove: WordEmbeddings) -> ContextualLSTM:
    model = ContextualLSTM(glove, device="cuda")
    model.to(device="cuda")

    logging.info("Created ContextualLSTM to train (device=%s)", model.device)

    return model


def create_optimizer(args, model: ContextualLSTM) -> Optimizer:
    # TODO: Exclude embedding weights from Adam
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta0, args.beta1),
        eps=args.eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )

    logging.info(
        "Created Adam optimizer (lr=%g, betas=(%g, %g), eps=%g, weight_decay=%g, amsgrad=%s)",
        args.lr,
        args.beta0,
        args.beta1,
        args.eps,
        args.weight_decay,
        args.amsgrad
    )

    return optimizer


def load_tweet_datasets(args, datasets, type: str, glove: WordEmbeddings) -> Dataset:
    loaded = []
    for d in datasets:
        logging.info("Loading known %ss from %s", type, d.name)
        loaded.append(load_tweets(d, glove))

    dataset = None
    if len(loaded) == 1:
        dataset = loaded[0]
    else:
        dataset = ConcatDataset(loaded)

    if args.max_tweets is not None:
        indices = random.sample(range(len(dataset)), args.max_tweets // 2)
        dataset = Subset(dataset, indices)

    logging.info("Loaded %d %s datasets with %d tweets", len(loaded), type, len(dataset))

    return dataset


def create_splits(args, type: str, data: Dataset) -> Splits:
    length = len(data)
    split_lengths = split_integers(length, (args.train_split, args.valid_split, args.test_split))

    logging.info(
        "Splitting %d %s tweets (train=%g, valid=%g, test=%g)",
        length,
        type,
        args.train_split,
        args.valid_split,
        args.test_split
    )

    splits = random_split(data, split_lengths)

    logging.info("Split %d %s tweets (train=%d, valid=%d, test=%d)", length, type, *split_lengths)

    return Splits(data, *splits)


def create_loader(args, human: DataLoader, bot: DataLoader, subset: str) -> DataLoader:
    human = SingleLabelDataset(human, NOT_BOT)
    bot = SingleLabelDataset(bot, BOT)

    dataset = ConcatDataset([human, bot])
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=args.batch_size, collate_fn=sentence_label_pad)

    logging.info("Created a %s DataLoader (len=%d, batch_size=%d)", subset, len(dataset), args.batch_size)
    return dataloader


def create_lr_scheduler(args, optimizer: Optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience)

    return scheduler


def create_evaluator(model: ContextualLSTM, cost: Module):
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            "loss": Loss(cost, output_transform=to_singleton_row),
            "accuracy": BinaryAccuracy(output_transform=to_singleton_row),
            "recall": Recall(average=True, output_transform=expand_binary_class),
            "precision": Precision(average=True, output_transform=expand_binary_class),
        }
    )

    evaluator._logger.setLevel(logging.WARNING)
    return evaluator


def create_trainer(
    args,
    model: ContextualLSTM,
    optimizer: Optimizer,
    cost: Module,
    evaluator: Engine,
    scheduler,
    training_data: DataLoader,
    validation_data: DataLoader
):
    model.train(True)

    trainer = ignite.engine.create_supervised_trainer(model, optimizer, cost, model.device)
    trainer.state = ignite.engine.State()

    @trainer.on(Events.COMPLETED)
    def finish_training(trainer: Engine):
        model.train(False)
        logging.info("Finished training and evaluation")

    @trainer.on(Events.STARTED)
    def init_metrics(trainer: Engine):
        trainer.state.training_metrics = Metrics([], [], [], [])
        trainer.state.validation_metrics = Metrics([], [], [], [])
        logging.info("Initialized metrics")

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(trainer: Engine):
        training_metrics = evaluator.run(training_data).metrics  # type: Dict[str, float]
        trainer.state.training_metrics.loss.append(training_metrics["loss"])
        trainer.state.training_metrics.accuracy.append(training_metrics["accuracy"])
        trainer.state.training_metrics.recall.append(training_metrics["recall"])
        trainer.state.training_metrics.precision.append(training_metrics["precision"])
        logging.info(
            "[%d / %d] Train: (loss=%.4f, accuracy=%.4f, recall=%.4f, precision=%.4f",
            trainer.state.epoch,
            trainer.state.max_epochs,
            training_metrics["loss"],
            training_metrics["accuracy"],
            training_metrics["recall"],
            training_metrics["precision"]
        )

        validation_metrics = evaluator.run(validation_data).metrics  # type: Dict[str, float]
        trainer.state.validation_metrics.loss.append(validation_metrics["loss"])
        trainer.state.validation_metrics.accuracy.append(validation_metrics["accuracy"])
        trainer.state.validation_metrics.recall.append(validation_metrics["recall"])
        trainer.state.validation_metrics.precision.append(validation_metrics["precision"])
        logging.info(
            "[%d / %d] Valid: (loss=%.4f, accuracy=%.4f, recall=%.4f, precision=%.4f",
            trainer.state.epoch,
            trainer.state.max_epochs,
            validation_metrics["loss"],
            validation_metrics["accuracy"],
            validation_metrics["recall"],
            validation_metrics["precision"]
        )
        scheduler.step(validation_metrics["loss"])

    timer = Timer(average=True)

    @trainer.on(Events.COMPLETED)
    def record_time(trainer: Engine):
        trainer.state.duration = timer.value()

    def score_function(trainer: Engine) -> float:
        return -trainer.state.validation_metrics.loss[-1]

    handler = EarlyStopping(patience=args.trainer_patience, score_function=score_function, trainer=trainer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    timer.attach(trainer, start=Events.STARTED, pause=Events.COMPLETED)
    trainer._logger.setLevel(logging.WARNING)
    return trainer


def main():
    logging.basicConfig(
        format='[%(levelname)s %(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    parser = build_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")

    if not torch.backends.cudnn.is_available():
        raise RuntimeError("CUDNN is required but not available")

    validate_args(args)
    cost = torch.nn.BCELoss()
    glove = load_glove(args)
    bots = load_tweet_datasets(args, args.bots, "bot", glove)
    humans = load_tweet_datasets(args, args.humans, "human", glove)

    bot_splits = create_splits(args, "bot", bots)
    human_splits = create_splits(args, "human", humans)

    training_data = create_loader(args, human_splits.training, bot_splits.training, "training")
    validation_data = create_loader(args, human_splits.validation, bot_splits.validation, "validation")
    testing_data = create_loader(args, human_splits.testing, bot_splits.testing, "testing")

    model = create_model(args, glove)
    optimizer = create_optimizer(args, model)
    lr_scheduler = create_lr_scheduler(args, optimizer)

    evaluator = create_evaluator(model, cost)
    trainer = create_trainer(
        args,
        model,
        optimizer,
        cost,
        evaluator,
        lr_scheduler,
        training_data,
        validation_data
    )

    train_result = trainer.run(training_data, max_epochs=args.max_epochs)  # type: State

    logging.info("Running trained model on test set")
    test_metrics = evaluator.run(testing_data).metrics  # type: dict
    logging.info("Finished running trained model on test set")

    logging.info("Results:")
    logging.info("    Time: %.2fs", train_result.duration)
    logging.info("    Epochs: %d / %d", train_result.epoch, train_result.max_epochs)
    logging.info("    Iterations: %d", train_result.iteration)
    logging.info("    Training:")
    logging.info("        Loss: %.4f", train_result.training_metrics.loss[-1])
    logging.info("        Accuracy: %.4f", train_result.training_metrics.accuracy[-1])
    logging.info("        Recall: %.4f", train_result.training_metrics.recall[-1])
    logging.info("        Precision: %.4f", train_result.training_metrics.precision[-1])
    logging.info("    Validation:")
    logging.info("        Loss: %.4f", train_result.validation_metrics.loss[-1])
    logging.info("        Accuracy: %.4f", train_result.validation_metrics.accuracy[-1])
    logging.info("        Recall: %.4f", train_result.validation_metrics.recall[-1])
    logging.info("        Precision: %.4f", train_result.validation_metrics.precision[-1])
    logging.info("    Testing:")
    logging.info("        Loss: %.4f", test_metrics['loss'])
    logging.info("        Accuracy: %.4f", test_metrics['accuracy'])
    logging.info("        Recall: %.4f", test_metrics['recall'])
    logging.info("        Precision: %.4f", test_metrics['precision'])
    logging.info("Accuracy: %.2f%% of all guesses were correct", test_metrics["accuracy"] * 100)
    logging.info("Recall: %.2f%% of guesses that should have identified bots did", test_metrics["recall"] * 100)
    logging.info("Precision: %.2f%% of 'bot' guesses were correct", test_metrics["precision"] * 100)

    save(model, args.output)
    logging.info("Saved trained model (minus embeddings) to %s", args.output.name)

if __name__ == '__main__':
    main()
