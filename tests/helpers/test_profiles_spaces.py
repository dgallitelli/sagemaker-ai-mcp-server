"""Tests for SageMaker AI User Profiles and Spaces."""

import pytest
from sagemaker_ai_mcp_server.helpers.profiles_spaces import list_spaces, list_user_profiles
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.profiles_spaces.get_sagemaker_client')
async def test_list_user_profiles(mock_get_sagemaker_client):
    """Test listing SageMaker AI User Profiles."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    mock_response = {
        'UserProfiles': [{'UserProfileName': 'test-user', 'UserProfileArn': 'arn:aws:...'}]
    }
    mock_client.list_user_profiles.return_value = mock_response
    profiles = await list_user_profiles()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_user_profiles.assert_called_once()
    expected = [{'UserProfileName': 'test-user', 'UserProfileArn': 'arn:aws:...'}]
    assert profiles == expected


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.profiles_spaces.get_sagemaker_client')
async def test_list_spaces(mock_get_sagemaker_client):
    """Test listing SageMaker AI Spaces."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    mock_response = {'Spaces': [{'SpaceName': 'test-space', 'SpaceId': 'space-id-123'}]}
    mock_client.list_spaces.return_value = mock_response
    spaces = await list_spaces()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_spaces.assert_called_once()
    expected = [{'SpaceName': 'test-space', 'SpaceId': 'space-id-123'}]
    assert spaces == expected
