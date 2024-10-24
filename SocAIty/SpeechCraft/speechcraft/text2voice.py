from typing import Union

from speechcraft.core.voice_embedding import VoiceEmbedding
from speechcraft.core.generation import SAMPLE_RATE, codec_decode, generate_coarse, generate_fine, generate_text_semantic
from speechcraft.supp.model_downloader import make_sure_models_are_downloaded
from speechcraft.settings import MODELS_DIR


def text2voice_advanced(
        text_prompt: str,
        speaker_name: str = None,
        semantic_temp=0.7,
        semantic_top_k=50,
        semantic_top_p=0.95,
        coarse_temp=0.7,
        coarse_top_k=50,
        coarse_top_p=0.95,
        fine_temp=0.5,
        use_semantic_history_prompt=True,
        use_coarse_history_prompt=True,
        use_fine_history_prompt=True,
        output_full=False
):
    """
    :param text_prompt:
    :param semantic_temp:
    :param semantic_top_k:
    :param semantic_top_p:
    :param coarse_temp:
    :param coarse_top_k:
    :param coarse_top_p:
    :param fine_temp:
    :param speaker_name:
    :param use_semantic_history_prompt:
    :param use_coarse_history_prompt:
    :param use_fine_history_prompt:
    :param output_full:
    :return:
    """

    # generation with more control
    x_semantic = generate_text_semantic(
        text_prompt,
        history_prompt=speaker_name if use_semantic_history_prompt else None,
        temp=semantic_temp,
        top_k=semantic_top_k,
        top_p=semantic_top_p,
    )

    x_coarse_gen = generate_coarse(
        x_semantic,
        history_prompt=speaker_name if use_coarse_history_prompt else None,
        temp=coarse_temp,
        top_k=coarse_top_k,
        top_p=coarse_top_p,
    )
    x_fine_gen = generate_fine(
        x_coarse_gen,
        history_prompt=speaker_name if use_fine_history_prompt else None,
        temp=fine_temp,
    )

    if output_full:
        full_generation = {
            'semantic_prompt': x_semantic,
            'coarse_prompt': x_coarse_gen,
            'fine_prompt': x_fine_gen,
        }
        return full_generation, codec_decode(x_fine_gen)
    return codec_decode(x_fine_gen)


def text2voice(
        text: str,
        voice: Union[str, VoiceEmbedding] = "en_speaker_3",
        semantic_temp=0.7,
        semantic_top_k=50,
        semantic_top_p=0.95,
        coarse_temp=0.7,
        coarse_top_k=50,
        coarse_top_p=0.95,
        fine_temp=0.5
) -> tuple:
    """
    :param text:
    :param voice: speaker name, path to the embedding or the VoiceEmbedding itself.
    :param semantic_temp:
    :param semantic_top_k:
    :param semantic_top_p:
    :param coarse_temp:
    :param coarse_top_k:
    :param coarse_top_p:
    :param fine_temp:
    :return:
    """

    make_sure_models_are_downloaded(install_path=MODELS_DIR)

    full_generation, audio_array = text2voice_advanced(
        text,
        semantic_temp=semantic_temp,
        semantic_top_k=semantic_top_k,
        semantic_top_p=semantic_top_p,
        coarse_temp=coarse_temp,
        coarse_top_k=coarse_top_k,
        coarse_top_p=coarse_top_p,
        fine_temp=fine_temp,
        speaker_name=voice,
        use_semantic_history_prompt=True,
        use_coarse_history_prompt=True,
        use_fine_history_prompt=True,
        output_full=True
    )
    return audio_array, SAMPLE_RATE
