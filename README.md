# FreshChain - Wallet-Free Blockchain Supply Chain

## Architecture Overview

FreshChain is a user-friendly blockchain dApp that provides transparent supply chain tracking without requiring MetaMask or crypto knowledge from retailers and consumers. The system uses a backend service with a system wallet to handle all blockchain transactions automatically.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Retailers     │    │   Backend       │    │   Blockchain    │
│   (No Wallet)   │    │   (System       │    │   (Shardeum)    │
│                 │    │    Wallet)      │    │                 │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Create batch  │───▶│ • Signs all     │───▶│ • Smart         │
│ • Update stages │    │   transactions  │    │   contracts     │
│ • Simple UI     │    │ • Handles gas   │    │ • Immutable     │
│ • No crypto     │    │ • API endpoints │    │   records       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Consumers     │
                    │   (No Wallet)   │
                    ├─────────────────┤
                    │ • Scan QR codes │
                    │ • View timeline │
                    │ • Read-only     │
                    │ • No setup      │
                    └─────────────────┘
```

## User Experience

**For Retailers:**
- No MetaMask installation required
- No wallet connection or gas fees
- Simple web interface for batch management
- Instant blockchain updates handled automatically

**For Consumers:**
- QR code scanning without any setup
- Complete supply chain visibility
- No technical knowledge required
- Instant verification results

## Data Storage

**On-Chain (Shardeum):**
- Batch creation records
- Stage update history
- Location tracking
- Timestamp verification
- Image analysis results

**Backend Service:**
- System wallet manages all transactions
- API endpoints for frontend communication
- Automatic gas fee handling
- Transaction signing and broadcasting

## Key Benefits

1. **User-Friendly**: No crypto knowledge required for retailers or consumers
2. **Instant Access**: No wallet setup or connection delays
3. **Real Blockchain**: Actual smart contract deployment with full transparency
4. **Professional UX**: Enterprise-grade interface hiding blockchain complexity
5. **Consumer Trust**: QR code verification without technical barriers

## Quick Setup

**Contract Details:**
- Network: Shardeum Testnet
- Chain ID: 8119
- Contract Address: `0x064e8D53bFF8023b0531FE845195c3741790870E`
- RPC URL: `https://api-mezame.shardeum.org`

**Development:**
```bash
npm install
npm run dev
```

**Access Points:**
- Retailer Dashboard: `http://localhost:3002/retailer`
- Consumer Audit: `http://localhost:3002/consumer-audit`
- Professional (MetaMask): `http://localhost:3002/professional`

## Architecture Benefits

This design prioritizes usability and real-world adoption while maintaining blockchain-backed transparency. By abstracting wallet complexity into a backend service, FreshChain becomes accessible to mainstream users while preserving all the benefits of blockchain verification.