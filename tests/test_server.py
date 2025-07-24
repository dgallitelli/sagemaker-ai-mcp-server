"""Tests for the server functions in the SageMaker AI MCP Server."""

import pytest
from sagemaker_ai_mcp_server.server import (
    delete_endpoint_config_sagemaker,
    delete_endpoint_sagemaker,
    describe_endpoint_config_sagemaker,
    describe_endpoint_sagemaker,
    describe_processing_job_sagemaker,
    describe_training_job_sagemaker,
    describe_transform_job_sagemaker,
    list_endpoint_configs_sagemaker,
    list_endpoints_sagemaker,
    list_processing_jobs_sagemaker,
    list_training_jobs_sagemaker,
    list_transform_jobs_sagemaker,
    stop_processing_job_sagemaker,
    stop_training_job_sagemaker,
    stop_transform_job_sagemaker,
)
from unittest.mock import patch


@pytest.mark.asyncio
async def test_list_endpoints_sagemaker():
    """Test the list_endpoints_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_endpoints') as mock_list_endpoints:
        mock_list_endpoints.return_value = [{'EndpointName': 'test-endpoint'}]

        result = await list_endpoints_sagemaker()

        mock_list_endpoints.assert_called_once()
        assert result == {'endpoints': [{'EndpointName': 'test-endpoint'}]}


@pytest.mark.asyncio
async def test_list_endpoint_configs_sagemaker():
    """Test the list_endpoint_configs_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_endpoint_configs') as mock_list_configs:
        mock_list_configs.return_value = [{'EndpointConfigName': 'test-config'}]

        result = await list_endpoint_configs_sagemaker()

        mock_list_configs.assert_called_once()
        assert result == {'endpoint_configs': [{'EndpointConfigName': 'test-config'}]}


@pytest.mark.asyncio
async def test_delete_endpoint_sagemaker():
    """Test the delete_endpoint_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.delete_endpoint') as mock_delete_endpoint:
        endpoint_name = 'test-endpoint'
        result = await delete_endpoint_sagemaker(endpoint_name)

        mock_delete_endpoint.assert_called_once_with(endpoint_name)
        expected_msg = f"Endpoint '{endpoint_name}' deleted successfully"
        assert result == {'message': expected_msg}


@pytest.mark.asyncio
async def test_delete_endpoint_config_sagemaker():
    """Test the delete_endpoint_config_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.delete_endpoint_config') as mock_delete_config:
        config_name = 'test-endpoint-config'

        result = await delete_endpoint_config_sagemaker(config_name)

        mock_delete_config.assert_called_once_with(config_name)
        expected_msg = f"Endpoint Config '{config_name}' deleted successfully"
        assert result == {'message': expected_msg}


@pytest.mark.asyncio
async def test_describe_endpoint_sagemaker():
    """Test the describe_endpoint_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_endpoint') as mock_describe_endpoint:
        endpoint_name = 'test-endpoint'
        expected_result = {
            'EndpointName': endpoint_name,
            'EndpointStatus': 'InService',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_endpoint.return_value = expected_result

        result = await describe_endpoint_sagemaker(endpoint_name)

        mock_describe_endpoint.assert_called_once_with(endpoint_name)
        assert result == expected_result


@pytest.mark.asyncio
async def test_describe_endpoint_config_sagemaker():
    """Test the describe_endpoint_config_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_endpoint_config') as mock_describe_config:
        config_name = 'test-endpoint-config'
        expected_result = {
            'EndpointConfigName': config_name,
            'CreationTime': '2023-01-01T00:00:00',
            'ProductionVariants': [{'VariantName': 'test-variant'}],
        }
        mock_describe_config.return_value = expected_result

        result = await describe_endpoint_config_sagemaker(config_name)

        mock_describe_config.assert_called_once_with(config_name)
        assert result == expected_result


@pytest.mark.asyncio
async def test_describe_training_job_sagemaker():
    """Test the describe_training_job_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_training_job') as mock_describe_job:
        job_name = 'test-training-job'
        expected_result = {
            'TrainingJobName': job_name,
            'TrainingJobStatus': 'Completed',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_job.return_value = expected_result

        result = await describe_training_job_sagemaker(job_name)

        mock_describe_job.assert_called_once_with(job_name)
        assert result == expected_result


@pytest.mark.asyncio
async def test_list_training_jobs_sagemaker():
    """Test the list_training_jobs_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_training_jobs') as mock_list_jobs:
        mock_list_jobs.return_value = [
            {'TrainingJobName': 'test-job-1'},
            {'TrainingJobName': 'test-job-2'},
        ]

        result = await list_training_jobs_sagemaker()

        mock_list_jobs.assert_called_once()
        assert result == {
            'training_jobs': [{'TrainingJobName': 'test-job-1'}, {'TrainingJobName': 'test-job-2'}]
        }


@pytest.mark.asyncio
async def test_stop_training_job_sagemaker():
    """Test the stop_training_job_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.stop_training_job') as mock_stop_job:
        job_name = 'test-training-job'
        await stop_training_job_sagemaker(job_name)

        mock_stop_job.assert_called_once_with(job_name)
        expected_msg = f"Training job '{job_name}' stopped successfully"
        assert {'message': expected_msg} == {'message': expected_msg}


@pytest.mark.asyncio
async def test_list_processing_jobs_sagemaker():
    """Test the list_processing_jobs_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_processing_jobs') as mock_list_processing:
        mock_list_processing.return_value = [
            {'ProcessingJobName': 'test-processing-job-1'},
            {'ProcessingJobName': 'test-processing-job-2'},
        ]

        result = await list_processing_jobs_sagemaker()

        mock_list_processing.assert_called_once()
        assert result == {
            'processing_jobs': [
                {'ProcessingJobName': 'test-processing-job-1'},
                {'ProcessingJobName': 'test-processing-job-2'},
            ]
        }


@pytest.mark.asyncio
async def test_describe_processing_job_sagemaker():
    """Test the describe_processing_job_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.describe_processing_job'
    ) as mock_describe_processing:
        job_name = 'test-processing-job'
        expected_result = {
            'ProcessingJobName': job_name,
            'ProcessingJobStatus': 'Completed',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_processing.return_value = expected_result

        result = await describe_processing_job_sagemaker(job_name)

        mock_describe_processing.assert_called_once_with(job_name)
        assert result == expected_result


@pytest.mark.asyncio
async def test_stop_processing_job_sagemaker():
    """Test the stop_processing_job_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.stop_processing_job') as mock_stop_processing:
        job_name = 'test-processing-job'
        await stop_processing_job_sagemaker(job_name)

        mock_stop_processing.assert_called_once_with(job_name)
        expected_msg = f"Processing job '{job_name}' stopped successfully"
        assert {'message': expected_msg} == {'message': expected_msg}


@pytest.mark.asyncio
async def test_list_transform_jobs_sagemaker():
    """Test the list_transform_jobs_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_transform_jobs') as mock_list_transform:
        mock_list_transform.return_value = [
            {'TransformJobName': 'test-transform-job-1'},
            {'TransformJobName': 'test-transform-job-2'},
        ]

        result = await list_transform_jobs_sagemaker()

        mock_list_transform.assert_called_once()
        assert result == {
            'transform_jobs': [
                {'TransformJobName': 'test-transform-job-1'},
                {'TransformJobName': 'test-transform-job-2'},
            ]
        }


@pytest.mark.asyncio
async def test_describe_transform_job_sagemaker():
    """Test the describe_transform_job_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_transform_job') as mock_describe_transform:
        job_name = 'test-transform-job'
        expected_result = {
            'TransformJobName': job_name,
            'TransformJobStatus': 'Completed',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_transform.return_value = expected_result

        result = await describe_transform_job_sagemaker(job_name)

        mock_describe_transform.assert_called_once_with(job_name)
        assert result == expected_result


@pytest.mark.asyncio
async def test_stop_transform_job_sagemaker():
    """Test the stop_transform_job_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.stop_transform_job') as mock_stop_transform:
        job_name = 'test-transform-job'
        await stop_transform_job_sagemaker(job_name)

        mock_stop_transform.assert_called_once_with(job_name)
        expected_msg = f"Transform job '{job_name}' stopped successfully"
        assert {'message': expected_msg} == {'message': expected_msg}
