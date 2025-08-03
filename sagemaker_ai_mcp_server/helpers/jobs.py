"""Helper Functions for SageMaker Jobs(Training, Processing, Transform, Inference Recommender)."""

from loguru import logger
from sagemaker_ai_mcp_server.helpers.utils import get_sagemaker_client
from typing import Any, Dict, List


async def list_training_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Training Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Training Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Training Jobs...')
    response = client.list_training_jobs()
    return response.get('TrainingJobSummaries', [])


async def list_processing_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Processing Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Processing Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Processing Jobs...')
    response = client.list_processing_jobs()
    return response.get('ProcessingJobSummaries', [])


async def list_transform_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Transform Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Transform Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Transform Jobs...')
    response = client.list_transform_jobs()
    return response.get('TransformJobSummaries', [])


async def list_inference_recommendations_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Inference Recommender Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Inference Recommender Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Inference Recommender Jobs...')
    response = client.list_inference_recommendations_jobs()
    return response.get('InferenceRecommendationsJobs', [])


async def list_inference_recommendations_job_steps(job_name: str) -> List[Dict[str, Any]]:
    """List steps for a specific SageMaker Inference Recommender Job.

    Args:
        job_name (str): The name of the SageMaker Inference Recommender Job.

    Returns:
        List[Dict[str, Any]]: A list of steps for the specified Inference
        Recommender Job.
    """
    client = get_sagemaker_client()
    logger.info(f'Listing steps for Inference Recommender Job: {job_name}')
    response = client.list_inference_recommendations_job_steps(JobName=job_name)
    return response.get('Steps', [])


async def describe_training_job(training_job_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Training Job."""
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Training Job: {training_job_name}')
    response = client.describe_training_job(TrainingJobName=training_job_name)
    return response


async def describe_processing_job(processing_job_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Processing Job.

    Args:
        processing_job_name (str): The name of the SageMaker Processing Job to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Processing Job.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Processing Job: {processing_job_name}')
    response = client.describe_processing_job(ProcessingJobName=processing_job_name)
    return response


async def describe_transform_job(transform_job_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Transform Job.

    Args:
        transform_job_name (str): The name of the SageMaker Transform Job to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Transform Job.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Transform Job: {transform_job_name}')
    response = client.describe_transform_job(TransformJobName=transform_job_name)
    return response


async def describe_inference_recommendations_job(job_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Inference Recommender Job.

    Args:
        job_name (str): The name of the SageMaker Inference Recommender Job
            to describe.

    Returns:
        Dict[str, Any]: The details of the specified Inference Recommender Job.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Inference Recommender Job: {job_name}')
    response = client.describe_inference_recommendations_job(JobName=job_name)
    return response


async def stop_training_job(training_job_name: str) -> None:
    """Stop a SageMaker Training Job.

    Args:
        training_job_name (str): The name of the SageMaker Training Job to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping SageMaker Training Job: {training_job_name}')
    client.stop_training_job(TrainingJobName=training_job_name)
    logger.info(f'Training Job {training_job_name} stopped successfully.')


async def stop_processing_job(processing_job_name: str) -> None:
    """Stop a SageMaker Processing Job.

    Args:
        processing_job_name (str): The name of the SageMaker Processing Job to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping SageMaker Processing Job: {processing_job_name}')
    client.stop_processing_job(ProcessingJobName=processing_job_name)
    logger.info(f'Processing Job {processing_job_name} stopped successfully.')


async def stop_transform_job(transform_job_name: str) -> None:
    """Stop a SageMaker Transform Job.

    Args:
        transform_job_name (str): The name of the SageMaker Transform Job to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping SageMaker Transform Job: {transform_job_name}')
    client.stop_transform_job(TransformJobName=transform_job_name)


async def stop_inference_recommendations_job(job_name: str) -> None:
    """Stop a SageMaker Inference Recommender Job.

    Args:
        job_name (str): The name of the SageMaker Inference Recommender Job
            to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping SageMaker Inference Recommender Job: {job_name}')
    client.stop_inference_recommendations_job(JobName=job_name)
    logger.info(f'Inference Recommender Job {job_name} stopped successfully.')
