"""The main file for the SageMaker AI MCP Server."""

from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from sagemaker_ai_mcp_server.helpers import (
    delete_endpoint,
    delete_endpoint_config,
    describe_endpoint,
    describe_endpoint_config,
    describe_training_job,
    list_endpoint_configs,
    list_endpoints,
    list_training_jobs,
    stop_training_job,
)
from typing import Annotated, Any, Dict, List


mcp = FastMCP(
    'sagemaker-ai-mcp-server',
    instructions="""
    SageMaker AI MCP Server provides tools to interact with Amazon SageMaker AI
    service.
    
    The server enables you to:
    - List SageMaker Endpoints
    - List SageMaker Endpoint Configurations
    - Delete SageMaker Endpoints
    - Delete SageMaker Endpoint Configurations
    - Describe SageMaker Endpoints
    - Describe SageMaker Endpoint Configurations
    - List SageMaker Training Jobs
    - Describe SageMaker Training Jobs
    - Stop SageMaker Training Jobs

    Use these tools to manage your SageMaker resources effectively.
    """,
    dependencies=[
        'pydantic',
        'loguru',
        'boto3',
    ],
)


@mcp.tool(name='list_endpoints_sagemaker', description='List all SageMaker Endpoints')
async def list_endpoints_sagemaker() -> Dict[str, List]:
    """Get a list of all SageMaker Endpoint.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Endpoints in your account
    in the current region. This is typically used first to see what endpoints
    are available before performing operations on them.

    ## Example

    ```python
    endpoints = await list_endpoints_sagemaker()
    print(endpoints)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'endpoints': A list of dictionaries, each representing a SageMaker
      Endpoint with its details.

    ## Returns
    A dictionary containing a list of SageMaker Endpoints.
    """
    try:
        endpoints = await list_endpoints()
        return {'endpoints': endpoints}
    except Exception as e:
        logger.error(f'Error listing SageMaker endpoints: {e}')
        raise ValueError(f'Failed to list endpoints: {e}')


@mcp.tool(
    name='list_endpoint_configs_sagemaker',
    description='List all SageMaker Endpoint Configurations',
)
async def list_endpoint_configs_sagemaker() -> Dict[str, List]:
    """Get a list of all SageMaker Endpoint Configurations.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Endpoint Configurations
    in your account in the current region. This helps you identify available
    endpoint configurations before performing operations on them.

    ## Example

    ```python
    configs = await list_endpoint_configs_sagemaker()
    print(configs)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'endpoint_configs': A list of dictionaries, each representing a SageMaker
      Endpoint Configuration with its details.

    ## Returns
    A dictionary containing a list of SageMaker Endpoint Configurations.
    """
    try:
        configs = await list_endpoint_configs()
        return {'endpoint_configs': configs}
    except Exception as e:
        logger.error(f'Error listing endpoint configurations: {e}')
        raise ValueError(f'Failed to list endpoint configs: {e}')


@mcp.tool(name='delete_endpoint_sagemaker', description='Delete a SageMaker Endpoint')
async def delete_endpoint_sagemaker(
    endpoint_name: Annotated[
        str, Field(description='The name of the SageMaker Endpoint to delete')
    ],
) -> Dict[str, str]:
    """Delete a specified SageMaker Endpoint.

    ## Usage

    Use this tool to delete a SageMaker Endpoint by providing its name.
    This is useful for cleaning up resources that are no longer needed.

    ## Example

    ```python
    result = await delete_endpoint_sagemaker(endpoint_name='my-endpoint')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await delete_endpoint(endpoint_name)
        return {'message': f"Endpoint '{endpoint_name}' deleted successfully"}
    except Exception as e:
        logger.error(f'Error deleting endpoint {endpoint_name}: {e}')
        raise ValueError(f'Failed to delete endpoint {endpoint_name}: {e}')


@mcp.tool(
    name='delete_endpoint_config_sagemaker',
    description='Delete a SageMaker Endpoint Configuration',
)
async def delete_endpoint_config_sagemaker(
    endpoint_config_name: Annotated[
        str, Field(description='The name of the SageMaker Endpoint Configuration to delete')
    ],
) -> Dict[str, str]:
    """Delete a specified SageMaker Endpoint Configuration.

    ## Usage

    Use this tool to delete a SageMaker Endpoint Configuration by providing
    its name. This is useful for cleaning up configurations that are no longer
    needed.

    ## Example

    ```python
    result = await delete_endpoint_config_sagemaker(endpoint_config_name='my-endpoint-config')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await delete_endpoint_config(endpoint_config_name)
        msg = f"Endpoint Config '{endpoint_config_name}' deleted successfully"
        return {'message': msg}
    except Exception as e:
        logger.error(f'Error deleting config {endpoint_config_name}: {e}')
        raise ValueError(f'Failed to delete config {endpoint_config_name}: {e}')


@mcp.tool(name='describe_endpoint_sagemaker', description='Describe a SageMaker Endpoint')
async def describe_endpoint_sagemaker(
    endpoint_name: Annotated[
        str, Field(description='The name of the SageMaker Endpoint to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Endpoint.

    ## Usage

    Use this tool to get detailed information about a SageMaker Endpoint by
    providing its name. This returns comprehensive information about the
    endpoint's configuration, status, and other details.

    ## Example

    ```python
    details = await describe_endpoint_sagemaker(endpoint_name='my-endpoint')
    print(details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker
    Endpoint.

    ## Returns
    A dictionary containing the endpoint details.
    """
    try:
        endpoint_details = await describe_endpoint(endpoint_name)
        return endpoint_details
    except Exception as e:
        logger.error(f'Error describing endpoint {endpoint_name}: {e}')
        raise ValueError(f'Failed to describe endpoint {endpoint_name}: {e}')


@mcp.tool(
    name='describe_endpoint_config_sagemaker',
    description='Describe a SageMaker Endpoint Configuration',
)
async def describe_endpoint_config_sagemaker(
    endpoint_config_name: Annotated[
        str, Field(description='The name of the SageMaker Endpoint Configuration to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Endpoint Configuration.

    ## Usage

    Use this tool to get detailed information about a SageMaker Endpoint
    Configuration by providing its name. This returns comprehensive information
    about the configuration details, including model specifications, instance
    types, and other configuration parameters.

    ## Example

    ```python
    config_details = await describe_endpoint_config_sagemaker(
        endpoint_config_name='my-endpoint-config'
    )
    print(config_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker
    Endpoint Configuration.

    ## Returns
    A dictionary containing the endpoint configuration details.
    """
    try:
        config_details = await describe_endpoint_config(endpoint_config_name)
        return config_details
    except Exception as e:
        logger.error(f'Error describing config {endpoint_config_name}: {e}')
        err_msg = f'Failed to describe config {endpoint_config_name}: {e}'
        raise ValueError(err_msg)


@mcp.tool(name='list_training_jobs_sagemaker', description='List SageMaker Training Jobs')
async def list_training_jobs_sagemaker() -> Dict[str, List]:
    """List all SageMaker Training Jobs.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Training Jobs in your
    account in the current region. This is typically used to see what training
    jobs are available before performing operations on them.

    ## Example

    ```python
    jobs = await list_training_jobs_sagemaker()
    print(jobs)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'training_jobs': A list of dictionaries, each representing a SageMaker
      Training Job with its details.

    ## Returns
    A dictionary containing a list of SageMaker Training Jobs.
    """
    try:
        jobs = await list_training_jobs()
        return {'training_jobs': jobs}
    except Exception as e:
        logger.error(f'Error listing training jobs: {e}')
        raise ValueError(f'Failed to list training jobs: {e}')


@mcp.tool(name='describe_training_job_sagemaker', description='Describe a SageMaker Training Job')
async def describe_training_job_sagemaker(
    training_job_name: Annotated[
        str, Field(description='The name of the SageMaker Training Job to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Training Job.

    ## Usage

    Use this tool to get detailed information about a SageMaker Training Job
    by providing its name. This returns comprehensive information about the
    job's configuration, status, and other details.

    ## Example

    ```python
    job_details = await describe_training_job_sagemaker(training_job_name='my-training-job')
    print(job_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker
    Training Job.

    ## Returns
    A dictionary containing the training job details.
    """
    try:
        job_details = await describe_training_job(training_job_name)
        return job_details
    except Exception as e:
        logger.error(f'Error describing training job {training_job_name}: {e}')
        raise ValueError(f'Failed to describe training job {training_job_name}: {e}')


@mcp.tool(name='stop_training_job_sagemaker', description='Stop a SageMaker Training Job')
async def stop_training_job_sagemaker(
    training_job_name: Annotated[
        str, Field(description='The name of the SageMaker Training Job to stop')
    ],
) -> Dict[str, str]:
    """Stop a specified SageMaker Training Job.

    ## Usage

    Use this tool to stop a SageMaker Training Job by providing its name.
    This is useful for terminating jobs that are no longer needed or are taking
    too long to complete.

    ## Example

    ```python
    result = await stop_training_job_sagemaker(training_job_name='my-training-job')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await stop_training_job(training_job_name)
        return {'message': f"Training Job '{training_job_name}' stopped successfully"}
    except Exception as e:
        logger.error(f'Error stopping training job {training_job_name}: {e}')
        raise ValueError(f'Failed to stop training job {training_job_name}: {e}')


def main():
    """Run the SageMaker AI MCP Server."""
    logger.info('Starting SageMaker AI MCP Server...')
    mcp.run()


if __name__ == '__main__':
    main()
