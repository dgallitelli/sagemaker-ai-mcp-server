"""Tests for SageMaker AI Pipelines."""

import pytest
from sagemaker_ai_mcp_server.helpers.pipelines import (
    delete_pipeline,
    describe_pipeline,
    describe_pipeline_definition_for_execution,
    describe_pipeline_execution,
    list_pipeline_execution_steps,
    list_pipeline_executions,
    list_pipeline_parameters_for_execution,
    list_pipelines,
    start_pipeline_execution,
    stop_pipeline_execution,
)
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.pipelines.get_sagemaker_client')
async def test_list_pipelines(mock_get_sagemaker_client):
    """Test listing SageMaker AI Pipelines."""
    mock_client = MagicMock()
    mock_client.list_pipelines.return_value = {
        'PipelineSummaries': [{'PipelineName': 'test-pipeline'}]
    }
    mock_get_sagemaker_client.return_value = mock_client
    pipelines = await list_pipelines()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_pipelines.assert_called_once()
    assert pipelines == [{'PipelineName': 'test-pipeline'}]


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.pipelines.get_sagemaker_client')
async def test_list_pipeline_executions(mock_get_sagemaker_client):
    """Test listing SageMaker AI Pipeline Executions."""
    mock_client = MagicMock()
    mock_client.list_pipeline_executions.return_value = {
        'PipelineExecutionSummaries': [
            {
                'PipelineExecutionArn': 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution'
            }
        ]
    }
    mock_get_sagemaker_client.return_value = mock_client
    executions = await list_pipeline_executions('test-pipeline')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_pipeline_executions.assert_called_once_with(PipelineName='test-pipeline')
    assert executions == [
        {
            'PipelineExecutionArn': 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution'
        }
    ]


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.pipelines.get_sagemaker_client')
async def test_list_pipeline_execution_steps(mock_get_sagemaker_client):
    """Test listing SageMaker AI Pipeline Execution Steps."""
    mock_client = MagicMock()
    mock_client.list_pipeline_execution_steps.return_value = {
        'PipelineExecutionSteps': [{'StepName': 'test-step', 'StepStatus': 'Succeeded'}]
    }
    mock_get_sagemaker_client.return_value = mock_client
    steps = await list_pipeline_execution_steps('test-execution')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_pipeline_execution_steps.assert_called_once_with(
        PipelineExecutionArn='test-execution'
    )
    assert steps == [{'StepName': 'test-step', 'StepStatus': 'Succeeded'}]


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.pipelines.get_sagemaker_client')
async def test_list_pipeline_parameters_for_execution(mock_get_sagemaker_client):
    """Test listing SageMaker AI Pipeline Parameters for Execution."""
    mock_client = MagicMock()
    mock_client.list_pipeline_parameters_for_execution.return_value = {
        'PipelineParameters': [{'Name': 'param1', 'Value': 'value1'}]
    }
    mock_get_sagemaker_client.return_value = mock_client
    parameters = await list_pipeline_parameters_for_execution('test-execution')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_pipeline_parameters_for_execution.assert_called_once_with(
        PipelineExecutionArn='test-execution'
    )
    assert parameters == [{'Name': 'param1', 'Value': 'value1'}]


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.pipelines.get_sagemaker_client')
async def test_describe_pipeline(mock_get_sagemaker_client):
    """Test describing a SageMaker AI Pipeline."""
    mock_client = MagicMock()
    expected_response = {'PipelineName': 'test-pipeline', 'PipelineStatus': 'Active'}
    mock_client.describe_pipeline.return_value = expected_response
    mock_get_sagemaker_client.return_value = mock_client
    response = await describe_pipeline('test-pipeline')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_pipeline.assert_called_once_with(PipelineName='test-pipeline')
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.pipelines.get_sagemaker_client')
async def test_describe_pipeline_definition_for_execution(mock_get_sagemaker_client):
    """Test describing a SageMaker AI Pipeline Definition for Execution."""
    mock_client = MagicMock()
    expected_response = {
        'PipelineDefinition': 'pipeline-definition-content',
        'PipelineDefinitionS3Location': {'Bucket': 'test-bucket', 'Key': 'test-key'},
    }
    mock_client.describe_pipeline_definition_for_execution.return_value = expected_response
    mock_get_sagemaker_client.return_value = mock_client
    response = await describe_pipeline_definition_for_execution('test-execution')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_pipeline_definition_for_execution.assert_called_once_with(
        PipelineExecutionArn='test-execution'
    )
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.pipelines.get_sagemaker_client')
async def test_describe_pipeline_execution(mock_get_sagemaker_client):
    """Test describing a SageMaker AI Pipeline Execution."""
    mock_client = MagicMock()
    expected_response = {
        'PipelineExecutionArn': 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution',
        'PipelineExecutionStatus': 'InProgress',
    }
    mock_client.describe_pipeline_execution.return_value = expected_response
    mock_get_sagemaker_client.return_value = mock_client
    response = await describe_pipeline_execution('test-execution')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_pipeline_execution.assert_called_once_with(
        PipelineExecutionArn='test-execution'
    )
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.pipelines.get_sagemaker_client')
async def test_start_pipeline_execution_without_parameters(mock_get_sagemaker_client):
    """Test starting a SageMaker AI Pipeline Execution without parameters."""
    mock_client = MagicMock()
    pipeline_arn = 'arn:aws:sagemaker:us-west-2:123456789012:'
    pipeline_path = 'pipeline/test-pipeline/execution/test-execution'
    expected_response = {'PipelineExecutionArn': pipeline_arn + pipeline_path}
    mock_client.start_pipeline_execution.return_value = expected_response
    mock_get_sagemaker_client.return_value = mock_client
    response = await start_pipeline_execution('test-pipeline')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.start_pipeline_execution.assert_called_once_with(
        PipelineName='test-pipeline', PipelineParameters=[]
    )
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.pipelines.get_sagemaker_client')
async def test_start_pipeline_execution_with_parameters(mock_get_sagemaker_client):
    """Test starting a SageMaker AI Pipeline Execution with parameters."""
    mock_client = MagicMock()
    pipeline_arn = 'arn:aws:sagemaker:us-west-2:123456789012:'
    pipeline_path = 'pipeline/test-pipeline/execution/test-execution'
    expected_response = {'PipelineExecutionArn': pipeline_arn + pipeline_path}
    mock_client.start_pipeline_execution.return_value = expected_response
    mock_get_sagemaker_client.return_value = mock_client
    pipeline_parameters = [
        {'Name': 'param1', 'Value': 'value1'},
        {'Name': 'param2', 'Value': 'value2'},
    ]
    response = await start_pipeline_execution('test-pipeline', pipeline_parameters)
    mock_get_sagemaker_client.assert_called_once()
    mock_client.start_pipeline_execution.assert_called_once_with(
        PipelineName='test-pipeline', PipelineParameters=pipeline_parameters
    )
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.pipelines.get_sagemaker_client')
async def test_stop_pipeline_execution(mock_get_sagemaker_client):
    """Test stopping a SageMaker AI Pipeline Execution."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    pipeline_arn = 'arn:aws:sagemaker:us-west-2:123456789012:'
    pipeline_path = 'pipeline/test-pipeline/execution/test-execution'
    execution_arn = pipeline_arn + pipeline_path
    await stop_pipeline_execution(execution_arn)
    mock_get_sagemaker_client.assert_called_once()
    mock_client.stop_pipeline_execution.assert_called_once_with(PipelineExecutionArn=execution_arn)


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.pipelines.get_sagemaker_client')
async def test_delete_pipeline(mock_get_sagemaker_client):
    """Test deleting a SageMaker AI Pipeline."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    await delete_pipeline('test-pipeline')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.delete_pipeline.assert_called_once_with(PipelineName='test-pipeline')
