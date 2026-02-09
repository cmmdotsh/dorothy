# Compatible with both Terraform and OpenTofu
# Use: tofu init && tofu apply

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment to use remote state
  # backend "s3" {
  #   bucket = "your-terraform-state-bucket"
  #   key    = "dorothy/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

# Default provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "dorothy"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ACM certificates for CloudFront must be in us-east-1
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"

  default_tags {
    tags = {
      Project     = "dorothy"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Look up Route53 zone by domain name
data "aws_route53_zone" "main" {
  count = var.use_route53 ? 1 : 0
  name  = var.zone_domain
}
