/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  env: {
    NEXT_PUBLIC_SHARDEUM_RPC: 'https://api-mezame.shardeum.org',
    NEXT_PUBLIC_CHAIN_ID: '8119',
    NEXT_PUBLIC_CONTRACT_ADDRESS: '0x064e8D53bFF8023b0531FE845195c3741790870E'
  }
}

module.exports = nextConfig