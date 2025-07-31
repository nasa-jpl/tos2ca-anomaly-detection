# Required Resources

TOS<sup>2</sup>CA was developed using Amazon Web Services (AWS).  Below is a list of services used while it was in development.

Generally you will need:
- Access to an S3 bucket where you can read and write data
- Access to a MySQL database that stores user input
- Access to a Redis/Elasticache memory store to temporarily house data that's being read/curated
- Access to AWS Secrets Manager to retrieve things like credentails, tokes, etc.
- Should have a [NASA Earthdata login](https://urs.earthdata.nasa.gov) to use any tools DAAC tools/applications
- A .netrc file with your NASA Earthdata login credentials
- Should have access to the ``us-west-2`` AWS region to access any NASA DAAC data over S3
- This code can be run on a single server or chunked and containerized and run in parallel
Additional information on specific AWS resources is below.

## Services

- **EC2 Server**: Used as a dev server for testing and hosting the TOS<sup>2</sup>CA website
- **RDS**: Used MySQL through the Relational Database Service as the backend for the website and to store job parameters; see the [database structure](../db/tosca_db.sql) for getting setup
- **S3**: Stores jobs outputs
- **Elastic Container Repository (ECR)**: Stores the containers; this is where Fargate grabs them and runs them in a serverless fashion
- **Elastic Container Service (ECS)**: Used Fargate and runs job chunks, using the ECR containers
- Secrets Manager: Stores information like passwords, resource names, etc. that are retried through the TOS<sup>2</sup>CA Python code 
- **Elasticache**:  Used ValKey to temporarily store PhDef and ForTraCC job chunked outputs
- **VPC**: Controles the virtual private cloud that all resources were contained in
- **IAM**: For managing access and permissions through the various resources
- **AWS Secrets**: For credential management; see [template](../templates/aws_secrets.json) for required fields

Reminder that all resources were setup in `us-west-2` so that they would have free access to NASA Earth Science data.
