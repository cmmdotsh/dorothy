# IAM user for deploy pipeline
resource "aws_iam_user" "deploy" {
  name = "dorothy-deploy"
  path = "/service/"
}

# Access key for the deploy user
resource "aws_iam_access_key" "deploy" {
  user = aws_iam_user.deploy.name
}

# Policy with minimal permissions for deploy
resource "aws_iam_user_policy" "deploy" {
  name = "dorothy-deploy-policy"
  user = aws_iam_user.deploy.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3Upload"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:GetObject",
          "s3:ListBucket",
        ]
        Resource = [
          aws_s3_bucket.site.arn,
          "${aws_s3_bucket.site.arn}/*",
        ]
      },
      {
        Sid    = "CloudFrontInvalidate"
        Effect = "Allow"
        Action = [
          "cloudfront:CreateInvalidation",
          "cloudfront:GetInvalidation",
          "cloudfront:ListInvalidations",
        ]
        Resource = aws_cloudfront_distribution.site.arn
      },
    ]
  })
}
