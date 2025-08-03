"""Helper Functions for SageMaker Pipelines"""

from loguru import logger
from sagemaker_ai_mcp_server.helpers.utils import get_sagemaker_client
from typing import Any, Dict, List


async def list_pipelines() -> List[Dict[str, Any]]:
    """List all SageMaker Pipelines.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Pipelines.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Pipelines...')
    response = client.list_pipelines()
    return response.get('PipelineSummaries', [])


async def list_pipeline_parameters_for_execution(
    pipeline_execution_arn: str,
) -> List[Dict[str, Any]]:
    """List parameters for a specific SageMaker Pipeline Execution.

    Args:
        pipeline_execution_arn (str): The ARN of the SageMaker Pipeline Execution.

    Returns:
        List[Dict[str, Any]]: A list of parameters for the specified Pipeline Execution.
    """
    client = get_sagemaker_client()
    logger.info(f'Listing parameters for Pipeline Execution: {pipeline_execution_arn}')
    response = client.list_pipeline_parameters_for_execution(
        PipelineExecutionArn=pipeline_execution_arn
    )
    return response.get('PipelineParameters', [])


async def list_pipeline_executions(pipeline_name: str) -> List[Dict[str, Any]]:
    """List all executions of a specific SageMaker Pipeline.

    Args:
        pipeline_name (str): The name of the SageMaker Pipeline.

    Returns:
        List[Dict[str, Any]]: A list of Pipeline Executions for the specified Pipeline.
    """
    client = get_sagemaker_client()
    logger.info(f'Listing executions for Pipeline: {pipeline_name}')
    response = client.list_pipeline_executions(PipelineName=pipeline_name)
    return response.get('PipelineExecutionSummaries', [])


async def list_pipeline_execution_steps(
    pipeline_execution_arn: str,
) -> List[Dict[str, Any]]:
    """List steps for a specific SageMaker Pipeline Execution.

    Args:
        pipeline_execution_arn (str): The ARN of the SageMaker Pipeline Execution.

    Returns:
        List[Dict[str, Any]]: A list of steps for the specified Pipeline Execution.
    """
    client = get_sagemaker_client()
    logger.info(f'Listing steps for Pipeline Execution: {pipeline_execution_arn}')
    response = client.list_pipeline_execution_steps(PipelineExecutionArn=pipeline_execution_arn)
    return response.get('PipelineExecutionSteps', [])


async def describe_pipeline(pipeline_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Pipeline.

    Args:
        pipeline_name (str): The name of the SageMaker Pipeline to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Pipeline.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Pipeline: {pipeline_name}')
    response = client.describe_pipeline(PipelineName=pipeline_name)
    return response


async def describe_pipeline_execution(
    pipeline_execution_arn: str,
) -> Dict[str, Any]:
    """Describe a specific SageMaker Pipeline Execution.

    Args:
        pipeline_execution_arn (str): The ARN of the SageMaker Pipeline Execution.

    Returns:
        Dict[str, Any]: The details of the specified Pipeline Execution.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing Pipeline Execution: {pipeline_execution_arn}')
    response = client.describe_pipeline_execution(PipelineExecutionArn=pipeline_execution_arn)
    return response


async def describe_pipeline_definition_for_execution(
    pipeline_execution_arn: str,
) -> Dict[str, Any]:
    """Describe the definition of a specific SageMaker Pipeline Execution.

    Args:
        pipeline_execution_arn (str): The ARN of the SageMaker Pipeline Execution.

    Returns:
        Dict[str, Any]: The definition of the specified Pipeline Execution.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing Pipeline Definition for Execution: {pipeline_execution_arn}')
    response = client.describe_pipeline_definition_for_execution(
        PipelineExecutionArn=pipeline_execution_arn
    )
    return response


async def start_pipeline_execution(
    pipeline_name: str,
    pipeline_parameters: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Start a new execution of a SageMaker Pipeline.

    Args:
        pipeline_name (str): The name of the SageMaker Pipeline.
        pipeline_parameters (List[Dict[str, Any]], optional): Parameters for the Pipeline Execution. Defaults to None.

    Returns:
        Dict[str, Any]: The details of the started Pipeline Execution.
    """
    client = get_sagemaker_client()
    logger.info(f'Starting Pipeline Execution for: {pipeline_name}')
    response = client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineParameters=pipeline_parameters or [],
    )
    return response


async def stop_pipeline_execution(pipeline_execution_arn: str) -> None:
    """Stop a specific SageMaker Pipeline Execution.

    Args:
        pipeline_execution_arn (str): The ARN of the SageMaker Pipeline Execution to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping Pipeline Execution: {pipeline_execution_arn}')
    client.stop_pipeline_execution(PipelineExecutionArn=pipeline_execution_arn)
    logger.info(f'Pipeline Execution {pipeline_execution_arn} stopped successfully.')


async def delete_pipeline(pipeline_name: str) -> None:
    """Delete a SageMaker Pipeline.

    Args:
        pipeline_name (str): The name of the SageMaker Pipeline to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker Pipeline: {pipeline_name}')
    client.delete_pipeline(PipelineName=pipeline_name)
    logger.info(f'Pipeline {pipeline_name} deleted successfully.')
