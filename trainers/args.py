from argparse import ArgumentParser, Namespace
from attr import attrs
from pathlib import Path


@attrs(frozen=True, slots=True, auto_attribs=True)
class MainArgs:
    datasets: str
    model_name: str
    resume_from: str
    no_train: bool
    test: bool

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        arg = parser.add_argument
        arg(
            "--datasets",
            type=str,
            required=True,
            help="Comma-separated list of datasets to use"
        )
        arg(
            "--model_name",
            type=str,
            help="The name of the model to load from HuggingFace or local "
            "storage."
        )
        arg(
            "--resume_from",
            type=str,
            help="The path from which to load the model and optimizer. Either "
            "this or the model_name must be provided."
        )
        arg(
            "--no_train",
            action="store_true",
            help="If true then don't train the model."
        )
        arg(
            "--test",
            action="store_true",
            help="If true then evaluate on the test set."
        )

    @staticmethod
    def from_args(args: Namespace) -> "MainArgs":
        return MainArgs(
            datasets=args.datasets,
            model_name=args.model_name,
            resume_from=args.resume_from,
            no_train=args.no_train,
            test=args.test,
        )


@attrs(frozen=True, slots=True, auto_attribs=True)
class LoggingArgs:
    val_frequency: int
    checkpoint_frequency: int
    output_dir: Path

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        arg = parser.add_argument
        arg(
            "--val_frequency",
            type=int,
            default=1000,
            help="How often to perform validation (in steps)",
        )
        arg(
            "--checkpoint_frequency",
            type=int,
            default=1000,
            help="How often to save checkpoints (in steps)",
        )
        arg(
            "--output_dir",
            type=Path,
            help="Where to store checkpoint files",
            required=True,
        )

    @staticmethod
    def from_args(args: Namespace) -> "LoggingArgs":
        return LoggingArgs(
            val_frequency=args.val_frequency,
            checkpoint_frequency=args.checkpoint_frequency,
            output_dir=args.output_dir,
        )


@attrs(frozen=True, slots=True, auto_attribs=True)
class TrainingArgs:
    learning_rate: float
    max_iterations: int
    tokenize_max_length: int
    generate_max_length: int
    train_batch_size: int
    val_batch_size: int
    gen_batch_size: int
    n_examples_prompt: int
    n_beams: int
    augmentation: bool
    augmentation_prob: float

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        arg = parser.add_argument
        arg(
            "--learning_rate",
            type=float,
            help="Learning rate",
            default=1e-5,
        )
        arg(
            "--max_iterations",
            type=int,
            help="How many iterations to train for",
            default=10000,
        )
        arg(
            "--tokenize_max_length",
            type=int,
            help="Tokenize max length",
            default=512,
        )
        arg(
            "--generate_max_length",
            type=int,
            help="Generate max length",
            default=512,
        )
        arg(
            "--train_batch",
            type=int,
            help="Training batch size (per GPU)",
            default=8,
        )
        arg(
            "--val_batch",
            type=int,
            help="Validation batch size (per GPU)",
            default=16,
        )
        arg(
            "--gen_batch",
            type=int,
            help="Generation batch size (per GPU)",
            default=4,
        )
        arg(
            "--n_examples_prompt",
            type=int,
            help="Number of examples to show in the prompt",
            default=0,
        )
        arg(
            "--n_beams",
            type=int,
            help="Number of beams to use when generating",
            default=3,
        )
        arg(
            "--augmentation",
            action="store_true",
            help="If true then augment data during training by corrupting the "
            "schemas given in the prompt and the target JSONs",
        )
        arg(
            "--augmentation_prob",
            type=float,
            help="The probability that a JSON will be corrupted during "
            "training",
            default=0.5,
        )

    @staticmethod
    def from_args(args: Namespace) -> "TrainingArgs":
        return TrainingArgs(
            learning_rate=args.learning_rate,
            max_iterations=args.max_iterations,
            tokenize_max_length=args.tokenize_max_length,
            generate_max_length=args.generate_max_length,
            train_batch_size=args.train_batch,
            val_batch_size=args.val_batch,
            gen_batch_size=args.gen_batch,
            n_examples_prompt=args.n_examples_prompt,
            n_beams=args.n_beams,
            augmentation=args.augmentation,
            augmentation_prob=args.augmentation_prob,
        )