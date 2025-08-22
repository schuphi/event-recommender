/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    domains: [
      'localhost',
      'cdn.eventbrite.com',
      'scontent.cdninstagram.com',
      'v16-webapp.tiktok.com',
      'images.unsplash.com'
    ],
    unoptimized: true
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.NEXT_PUBLIC_API_URL + '/:path*'
      }
    ]
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_APP_NAME: 'Copenhagen Events',
    NEXT_PUBLIC_APP_VERSION: '1.0.0'
  },
  experimental: {
    appDir: true
  }
}

module.exports = nextConfig