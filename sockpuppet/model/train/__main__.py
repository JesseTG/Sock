import argparse
from argparse import FileType, ArgumentParser, ArgumentTypeError
import logging
import math

import torch
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset
from torch.nn import Module

import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine, State
from ignite.handlers import EarlyStopping, Timer
from ignite.metrics import Loss, BinaryAccuracy, Precision, Recall

from sockpuppet.model.data import WordEmbeddings
from sockpuppet.model.nn import ContextualLSTM
from sockpuppet.utils import Metrics, Splits, to_singleton_row, expand_binary_class

NOT_BOT = 0
BOT = 1


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


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Train a model"
    )

    # parser.add_argument_group("Data")
    parser.add_argument(
        "--glove",
        help="The word vector embeddings to use",
        metavar="path",
        type=FileType('r'),
        required=True
    )

    # TODO: Validate arguments with utility functions
    parser.add_argument(
        "--bot",
        required=True
    )

    parser.add_argument(
        "--human",
        required=True
    )

    parser.add_argument(
        "--output",
        type=FileType("wb"),
        required=True
    )

    # parser.add_argument_group("Optimizer Hyperparameters")

    parser.add_argument(
        "--lr",
        help="Learning rate (default: %(default)s)",
        type=positive_finite_float,
        default=1e-3,
        metavar="lr"
    )

    parser.add_argument(
        "--eps",
        help="Term added to the denominator to improve numerical stability (default: %(default)s)",
        type=positive_finite_float,
        default=1e-8,
        metavar="e"
    )

    parser.add_argument(
        "--beta0",
        help="First coefficient used for computing running averages of gradient and its square (default: %(default)s)",
        type=positive_finite_float,
        default=0.9,
        metavar="b0"
    )

    parser.add_argument(
        "--beta1",
        help="Second coefficient used for computing running averages of gradient and its square (default: %(default)s)",
        type=positive_finite_float,
        default=0.999,
        metavar="b1"
    )

    parser.add_argument(
        "--weight-decay",
        help="Weight decay (L2 penalty) (default: %(default)s)",
        type=nonzero_finite_float,
        default=0.0,
        metavar="wd"
    )

    parser.add_argument(
        "--amsgrad",
        help="Whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond (default: %(default)s)",
        action="store_true"
    )

    parser.add_argument(
        "--max-epochs",
        type=positive_int,
        default=50
    )

    parser.add_argument(
        "--trainer-patience",
        type=positive_int,
        default=10
    )

    parser.add_argument(
        "--lr-patience",
        type=positive_int,
        default=3
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=500
    )

    parser.add_argument(
        "--train-split",
        type=positive_finite_float,
        default=0.5
    )

    parser.add_argument(
        "--valid-split",
        type=positive_finite_float,
        default=0.2
    )

    parser.add_argument(
        "--test-split",
        type=positive_finite_float,
        default=0.3
    )

    # --output
    # --lr-scheduler
    # --params
    # --trainer type
    # --nbc path
    # --538 path
    return parser


def validate_args(args):
    if args.beta0 >= args.beta1:
        raise ArgumentTypeError(f"{args.beta0} is not less than {args.beta1}")

    if args.train_split + args.valid_split + args.test_split != 1.0:
        raise ArgumentTypeError(f"{args.train_split}, {args.valid_split}, and {args.test_split} do not add to 1")


def load_glove(args) -> WordEmbeddings:
    embeddings = WordEmbeddings(args.glove, device="cuda")

    logging.info("Loaded GloVe embeddings from %s (dim=%d)", args.glove, embeddings.dim)

    return embeddings


def create_model(args, glove: WordEmbeddings) -> ContextualLSTM:
    model = ContextualLSTM(glove, device="cuda")
    model.to(device="cuda")

    return model


def create_optimizer(args, model: ContextualLSTM) -> Optimizer:
    # TODO: Exclude embedding weights from Adam
    return Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta0, args.beta1),
        eps=args.eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )


def load_bots(args, glove: WordEmbeddings):
    pass


def load_humans(args, glove: WordEmbeddings):
    pass


def create_splits(args, bots: Dataset, humans: Dataset):
    pass


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

    return evaluator


def create_trainer(args, model: ContextualLSTM, optimizer: Optimizer, cost: Module, evaluator: Engine):
    model.train(True)

    trainer = ignite.engine.create_supervised_trainer(model, optimizer, cost, model.device)
    trainer.state = ignite.engine.State()

    @trainer.on(Events.STARTED)
    def set_model_in_state(trainer):
        trainer.state.model = model
        trainer.state.cost = cost
        logging.info("Initialized trainer model and cost function")

    @trainer.on(Events.COMPLETED)
    def finish_training(trainer):
        trainer.state.model.train(False)
        logging.info("Finished training and evaluation")

    set_model_in_state(trainer)

    @trainer.on(Events.STARTED)
    def init_metrics(trainer: Engine):
        trainer.state.training_metrics = Metrics([], [], [], [])
        trainer.state.validation_metrics = Metrics([], [], [], [])
        logging.info("Initialized metrics")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch(trainer: Engine):
        logging.info(f"Completed an epoch")

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(trainer: Engine):
        logging.info("Validating on training data")
        training_metrics = evaluator.run(training_data).metrics  # type: Dict[str, float]
        trainer.state.training_metrics.loss.append(training_metrics["loss"])
        trainer.state.training_metrics.accuracy.append(training_metrics["accuracy"])
        trainer.state.training_metrics.recall.append(training_metrics["recall"])
        trainer.state.training_metrics.precision.append(training_metrics["precision"])

        logging.info("Validating on validation data")
        validation_metrics = evaluator.run(validation_data).metrics  # type: Dict[str, float]
        trainer.state.validation_metrics.loss.append(validation_metrics["loss"])
        trainer.state.validation_metrics.accuracy.append(validation_metrics["accuracy"])
        trainer.state.validation_metrics.recall.append(validation_metrics["recall"])
        trainer.state.validation_metrics.precision.append(validation_metrics["precision"])
        scheduler.step(validation_metrics["loss"])

    timer = Timer(average=True)

    @trainer.on(Events.COMPLETED)
    def record_time(trainer: Engine):
        trainer.state.duration = timer.value()

    def score_function(trainer: Engine) -> float:
        return -trainer.state.validation_metrics.loss[-1]

    handler = EarlyStopping(patience=args.lr_patience, score_function=score_function, trainer=trainer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    timer.attach(trainer, start=Events.STARTED, pause=Events.COMPLETED)

    return trainer


def main():
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    # create optimizer
    # create_supervised_trainer
    # create_supervised_evaluator
    # define event handlers
    # create checkpointer
    # register event handlers
    # run trainer

    parser = build_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")

    if not torch.backends.cudnn.is_available():
        raise RuntimeError("CUDNN is required but not available")

    validate_args(args)
    cost = torch.nn.BCELoss()
    glove = load_glove(args)
    model = create_model(args, glove)
    optimizer = create_optimizer(args, model)

if __name__ == '__main__':
    main()

# input: training parameters
# output: a model
# in a program, i/o is done with python objects
# in a cli program, i/o is done with files
