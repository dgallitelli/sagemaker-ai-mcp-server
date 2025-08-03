"""Helper Functions for SageMaker Model Cards."""

from loguru import logger
from sagemaker_ai_mcp_server.helpers.utils import get_sagemaker_client
from typing import Any, Dict, List


async def list_model_cards() -> List[Dict[str, Any]]:
    """List all SageMaker Model Cards.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Model Cards.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Model Cards...')
    response = client.list_model_cards()
    return response.get('ModelCardSummaries', [])


async def list_model_card_export_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Model Card Export Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Model Card Export Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Model Card Export Jobs...')
    response = client.list_model_card_export_jobs()
    return response.get('ModelCardExportJobSummaries', [])


async def list_model_card_versions(model_card_name: str) -> List[Dict[str, Any]]:
    """List all versions of a SageMaker Model Card.

    Args:
        model_card_name (str): The name of the SageMaker Model Card.

    Returns:
        List[Dict[str, Any]]: A list of versions of the specified Model Card.
    """
    client = get_sagemaker_client()
    logger.info(f'Listing versions for Model Card: {model_card_name}')
    response = client.list_model_card_versions(ModelCardName=model_card_name)
    return response.get('ModelCardVersionSummaryList', [])


async def describe_model_card(model_card_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Model Card.

    Args:
        model_card_name (str): The name of the SageMaker Model Card to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Model Card.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Model Card: {model_card_name}')
    response = client.describe_model_card(ModelCardName=model_card_name)
    return response


async def delete_model_card(model_card_name: str) -> None:
    """Delete a SageMaker Model Card.

    Args:
        model_card_name (str): The name of the SageMaker Model Card to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker Model Card: {model_card_name}')
    client.delete_model_card(ModelCardName=model_card_name)
    logger.info(f'Model Card {model_card_name} deleted successfully.')
