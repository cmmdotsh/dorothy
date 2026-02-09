# Origin Access Control for S3
resource "aws_cloudfront_origin_access_control" "site" {
  name                              = "${var.bucket_name}-oac"
  description                       = "OAC for Dorothy static site"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# CloudFront distribution
resource "aws_cloudfront_distribution" "site" {
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  aliases             = [var.domain_name]
  price_class         = "PriceClass_100" # US, Canada, Europe only (cheapest)
  comment             = "Dorothy - Newspaper of Averages"

  origin {
    domain_name              = aws_s3_bucket.site.bucket_regional_domain_name
    origin_id                = "S3-${var.bucket_name}"
    origin_access_control_id = aws_cloudfront_origin_access_control.site.id
  }

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${var.bucket_name}"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 300  # 5 minutes for HTML
    max_ttl                = 3600 # 1 hour max
    compress               = true

    function_association {
      event_type   = "viewer-request"
      function_arn = aws_cloudfront_function.rewrite_uri.arn
    }
  }

  # Cache static assets longer
  ordered_cache_behavior {
    path_pattern     = "/static/*"
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${var.bucket_name}"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 86400    # 1 day minimum
    default_ttl            = 604800   # 1 week
    max_ttl                = 31536000 # 1 year
    compress               = true
  }

  # Custom error responses for SPA-like behavior
  custom_error_response {
    error_code         = 403
    response_code      = 404
    response_page_path = "/index.html"
  }

  custom_error_response {
    error_code         = 404
    response_code      = 404
    response_page_path = "/index.html"
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.site.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  depends_on = [aws_acm_certificate_validation.site]
}

# Route53 A record pointing to CloudFront
resource "aws_route53_record" "site" {
  count = var.use_route53 ? 1 : 0

  zone_id = data.aws_route53_zone.main[0].zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.site.domain_name
    zone_id                = aws_cloudfront_distribution.site.hosted_zone_id
    evaluate_target_health = false
  }
}

# AAAA record for IPv6
resource "aws_route53_record" "site_ipv6" {
  count = var.use_route53 ? 1 : 0

  zone_id = data.aws_route53_zone.main[0].zone_id
  name    = var.domain_name
  type    = "AAAA"

  alias {
    name                   = aws_cloudfront_distribution.site.domain_name
    zone_id                = aws_cloudfront_distribution.site.hosted_zone_id
    evaluate_target_health = false
  }
}
