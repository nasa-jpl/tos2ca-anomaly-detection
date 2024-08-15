import boto3
import json
import base64
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from json.decoder import JSONDecodeError

def get_secret(secret_name, region_name):
    """
    Function to retrieve information from AWS Secrets Manager
    :param secret_name: name of the secret to be retrieved
    :type secret_name: str
    :param region_name: name of the AWS region (like 'us-west-2')
    :type region_name: str
    :return secret: the secret information
    :rtype secret: dict
    """

    # Create a Secrets Manager client
    client = boto3.client(service_name='secretsmanager', region_name=region_name)

    try:
        # Get the secret value
        response = client.get_secret_value(SecretId=secret_name)

        # Get the value from the response
        if 'SecretString' in response:
            secret = response['SecretString']
        else: 
            # May be binary
            secret = base64.b64decode(response['SecretBinary'])

        #Return the code as a dictionary
        return json.loads(secret)
    except NoCredentialsError:
        print("No Credentials supplied")
        return None
    except PartialCredentialsError:
        print("Incomplete Credentials supplied")
        return None
    except JSONDecodeError:
        return response['SecretString'] # The secret itself is not JSON.
    except Exception as e:
        print("Error getting secret: {e}")
        return None