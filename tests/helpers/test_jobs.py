"""Tests for SageMaker AI Jobs (Training, Processing, Transform, Inference Recommender)."""

import pytest
from sagemaker_ai_mcp_server.helpers.jobs import (
    describe_inference_recommendations_job,
    describe_processing_job,
    describe_training_job,
    describe_transform_job,
    list_inference_recommendations_job_steps,
    list_inference_recommendations_jobs,
    list_processing_jobs,
    list_training_jobs,
    list_transform_jobs,
    stop_inference_recommendations_job,
    stop_processing_job,
    stop_training_job,
    stop_transform_job,
)
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_list_training_jobs(mock_get_sagemaker_client):
    """Test listing SageMaker AI Training Jobs."""
    mock_client = MagicMock()
    mock_client.list_training_jobs.return_value = {
        'TrainingJobSummaries': [{'TrainingJobName': 'test-job'}]
    }
    mock_get_sagemaker_client.return_value = mock_client
    jobs = await list_training_jobs()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_training_jobs.assert_called_once()
    assert jobs == [{'TrainingJobName': 'test-job'}]


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_list_processing_jobs(mock_get_sagemaker_client):
    """Test listing SageMaker AI Processing Jobs."""
    mock_client = MagicMock()
    mock_client.list_processing_jobs.return_value = {
        'ProcessingJobSummaries': [{'ProcessingJobName': 'test-processing-job'}]
    }
    mock_get_sagemaker_client.return_value = mock_client
    jobs = await list_processing_jobs()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_processing_jobs.assert_called_once()
    assert jobs == [{'ProcessingJobName': 'test-processing-job'}]


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_list_transform_jobs(mock_get_sagemaker_client):
    """Test listing SageMaker AI Transform Jobs."""
    mock_client = MagicMock()
    mock_client.list_transform_jobs.return_value = {
        'TransformJobSummaries': [{'TransformJobName': 'test-transform-job'}]
    }
    mock_get_sagemaker_client.return_value = mock_client
    jobs = await list_transform_jobs()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_transform_jobs.assert_called_once()
    assert jobs == [{'TransformJobName': 'test-transform-job'}]


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_list_inference_recommendations_jobs(mock_get_sagemaker_client):
    """Test listing SageMaker AI Inference Recommendations Jobs."""
    mock_client = MagicMock()
    mock_client.list_inference_recommendations_jobs.return_value = {
        'InferenceRecommendationsJobs': [
            {'JobName': 'test-job-1', 'Status': 'Completed'},
            {'JobName': 'test-job-2', 'Status': 'InProgress'},
        ]
    }
    mock_get_sagemaker_client.return_value = mock_client
    result = await list_inference_recommendations_jobs()
    assert len(result) == 2
    assert result[0]['JobName'] == 'test-job-1'
    assert result[1]['JobName'] == 'test-job-2'
    mock_client.list_inference_recommendations_jobs.assert_called_once()


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_list_inference_recommendations_job_steps(mock_get_sagemaker_client):
    """Test listing steps for a SageMaker AI Inference Recommendations Job."""
    job_name = 'test-job'
    mock_client = MagicMock()
    mock_client.list_inference_recommendations_job_steps.return_value = {
        'Steps': [
            {'StepName': 'step-1', 'Status': 'Completed'},
            {'StepName': 'step-2', 'Status': 'InProgress'},
        ]
    }
    mock_get_sagemaker_client.return_value = mock_client
    result = await list_inference_recommendations_job_steps(job_name)
    assert len(result) == 2
    assert result[0]['StepName'] == 'step-1'
    assert result[1]['StepName'] == 'step-2'
    mock_client.list_inference_recommendations_job_steps.assert_called_once_with(JobName=job_name)


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_describe_training_job(mock_get_sagemaker_client):
    """Test describing a SageMaker AI Training Job."""
    mock_client = MagicMock()
    expected_response = {'TrainingJobName': 'test-job', 'TrainingJobStatus': 'Completed'}
    mock_client.describe_training_job.return_value = expected_response
    mock_get_sagemaker_client.return_value = mock_client
    response = await describe_training_job('test-job')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_training_job.assert_called_once_with(TrainingJobName='test-job')
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_describe_processing_job(mock_get_sagemaker_client):
    """Test describing a SageMaker AI Processing Job."""
    mock_client = MagicMock()
    expected_response = {
        'ProcessingJobName': 'test-processing-job',
        'ProcessingJobStatus': 'Completed',
    }
    mock_client.describe_processing_job.return_value = expected_response
    mock_get_sagemaker_client.return_value = mock_client
    response = await describe_processing_job('test-processing-job')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_processing_job.assert_called_once_with(
        ProcessingJobName='test-processing-job'
    )
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_describe_transform_job(mock_get_sagemaker_client):
    """Test describing a SageMaker AI Transform Job."""
    mock_client = MagicMock()
    expected_response = {
        'TransformJobName': 'test-transform-job',
        'TransformJobStatus': 'Completed',
    }
    mock_client.describe_transform_job.return_value = expected_response
    mock_get_sagemaker_client.return_value = mock_client
    response = await describe_transform_job('test-transform-job')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_transform_job.assert_called_once_with(
        TransformJobName='test-transform-job'
    )
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_describe_inference_recommendations_job(mock_get_sagemaker_client):
    """Test describing a SageMaker AI Inference Recommendations Job."""
    job_name = 'test-job'
    mock_client = MagicMock()
    mock_client.describe_inference_recommendations_job.return_value = {
        'JobName': job_name,
        'Status': 'Completed',
        'JobType': 'Default',
        'CreationTime': '2023-01-01T00:00:00.000Z',
    }
    mock_get_sagemaker_client.return_value = mock_client
    result = await describe_inference_recommendations_job(job_name)
    assert result['JobName'] == job_name
    assert result['Status'] == 'Completed'
    assert result['JobType'] == 'Default'
    mock_client.describe_inference_recommendations_job.assert_called_once_with(JobName=job_name)


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_stop_training_job(mock_get_sagemaker_client):
    """Test stopping a SageMaker AI Training Job."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    await stop_training_job('test-job')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.stop_training_job.assert_called_once_with(TrainingJobName='test-job')


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_stop_processing_job(mock_get_sagemaker_client):
    """Test stopping a SageMaker AI Processing Job."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    await stop_processing_job('test-processing-job')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.stop_processing_job.assert_called_once_with(
        ProcessingJobName='test-processing-job'
    )


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_stop_transform_job(mock_get_sagemaker_client):
    """Test stopping a SageMaker AI Transform Job."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    await stop_transform_job('test-transform-job')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.stop_transform_job.assert_called_once_with(TransformJobName='test-transform-job')


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.jobs.get_sagemaker_client')
async def test_stop_inference_recommendations_job(mock_get_sagemaker_client):
    """Test stopping a SageMaker AI Inference Recommendations Job."""
    job_name = 'test-job'
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    await stop_inference_recommendations_job(job_name)
    mock_client.stop_inference_recommendations_job.assert_called_once_with(JobName=job_name)
