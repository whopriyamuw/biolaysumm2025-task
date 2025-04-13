from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from typing import Mapping

from torchtune.data import Message, InputOutputToMessages
from torchtune.datasets import samsum_dataset, PackedDataset, SFTDataset

__all__ = ["plos_dataset", "elife_dataset"]

from torchtune.modules.transforms.tokenizers import ModelTokenizer

_biolaysumm_dataset = partial(samsum_dataset, column_map={})
plos_dataset = partial(_biolaysumm_dataset, source="BioLaySumm/BioLaySumm2025-PLOS")
elife_dataset = partial(_biolaysumm_dataset, source="BioLaySumm/BioLaySumm2025-eLife")


def biolaysumm_dataset(
        tokenizer: ModelTokenizer,
        *,
        source: str = "BioLaySumm/BioLaySumm2025-PLOS",
        column_map: Optional[Dict[str, str]] = None,
        train_on_input: bool = False,
        new_system_prompt: Optional[str] = None,
        packed: bool = False,
        filter_fn: Optional[Callable] = None,
        split: str = "train",
        **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Support for summarization datasets and their variants from Hugging Face Datasets.
    An example is the `SAMsum dataset <https://huggingface.co/datasets/samsum>`_.

    It is recommended to configure the tokenizer with the :class:`~torchtune.data.SummarizeTemplate`
    in conjunction with this dataset.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``False`` by default
    - If ``train_on_input`` is True, the prompt is used during training and
    contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text"), pass
            in the filepath in ``data_files``, and set ``split="train"``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details. Default is ``Samsung/samsum``.
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns in the message transform
            :class:`~torchtune.data.InputOutputToMessages` to the new column names in the dataset. Keys should
            be "input" and "output" and values should be the actual column names. If None, use
            the default column names ``{"input": "dialogue", "output": "summary"}`` in ``Samsung/samsum``.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        new_system_prompt (Optional[str]): if specified, prepend a system message to every sample. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        packed (bool): Whether or not to pack the dataset to tokenizer's ``max_seq_len`` prior to training. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Returns:
        Union[SFTDataset, PackedDataset]: dataset configured with source data and template

    Raises:
        ValueError: If ``packed=True`` and ``tokenizer.max_seq_len`` is not set.

    Example:
        >>> samsum_ds = samsum_dataset(model_transform=tokenizer)
        >>> for batch in Dataloader(samsum_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """
    column_map = column_map or {"input": "article", "output": "summary", "title": "title"}

    message_transform = BLSToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )
    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        split=split,
        filter_fn=filter_fn,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds


class BLSToMessages(InputOutputToMessages):
    """
    Message transform class that converts a single sample with "input" and "output" fields,
    (or equivalent fields specified in column_map) to user and assistant messages,
    respectively. This is useful for datasets that have two columns, one containing
    the user prompt string and the other containing the model response string::

        |  input          |  output          |
        |-----------------|------------------|
        | "user prompt"   | "model response" |

    Args:
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is False.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "input"
            and "output" column names to the actual column names in the dataset. Keys should
            be "input" and "output" and values should be the actual column names. Default is None,
            keeping the default "input" and "output" column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.
        image_dir (Optional[Path]): path to the directory containing the images that is prepended to all image
            paths in the dataset. For example, if ``image_dir="/home/user/dataset/"` and the sample image path
            was ``"images/1.jpg"``, the final image path that will be loaded is ``"/home/user/dataset/images/1.jpg"``.
            If None, assume images are available in current working directory or are located
            on a remote url. For text-only, leave as None. Default is None.

    Raises:
        ValueError:
            If ``column_map`` is provided and ``input`` not in ``column_map``, or
                ``output`` not in ``column_map``, **or**
            if ``image_dir`` is provided but ``image`` not in ``column_map``.
    """

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        article = (f"Title: {sample[self.column_map['title']]}\n"
                   f"Content: {sample[self.column_map['input']]}")
        content = [{"type": "text", "content": article}]
        output_content = [
            {"type": "text", "content": sample[self.column_map["output"]]}
        ]

        messages = [
            Message(
                role="user",
                content=content,
                masked=not self.train_on_input,
                eot=True,
            ),
            Message(
                role="assistant",
                content=output_content,
                masked=False,
                eot=True,
            ),
        ]
        if self.new_system_prompt is not None:
            messages = [
                           Message(
                               role="system", content=self.new_system_prompt, masked=True, eot=True
                           )
                       ] + messages
        return {"messages": messages}
