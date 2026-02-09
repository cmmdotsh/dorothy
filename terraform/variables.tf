variable "aws_region" {
  description = "AWS region for resources (S3 bucket location)"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "prod"
}

variable "domain_name" {
  description = "Domain name for the site"
  type        = string
  default     = "dorothy.cmm.sh"
}

variable "bucket_name" {
  description = "S3 bucket name for static site"
  type        = string
  default     = "dorothy-cmm-sh"
}

# Set to true to use Route53 for DNS (looks up zone automatically)
variable "use_route53" {
  description = "Use Route53 for DNS management"
  type        = bool
  default     = true
}

# Parent domain for zone lookup
variable "zone_domain" {
  description = "Parent domain for Route53 zone lookup"
  type        = string
  default     = "cmm.sh"
}
