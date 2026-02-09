# CloudFront Function to handle directory index rewrites
# Converts /path → /path/index.html for S3 compatibility

resource "aws_cloudfront_function" "rewrite_uri" {
  name    = "dorothy-uri-rewrite"
  runtime = "cloudfront-js-2.0"
  publish = true
  comment = "Rewrite URIs to append index.html for directory paths"

  code = <<-EOF
    function handler(event) {
      var request = event.request;
      var uri = request.uri;

      // If URI ends with '/', append index.html
      if (uri.endsWith('/')) {
        request.uri += 'index.html';
      }
      // If URI doesn't have a file extension, append /index.html
      else if (!uri.includes('.')) {
        request.uri += '/index.html';
      }

      return request;
    }
  EOF
}
