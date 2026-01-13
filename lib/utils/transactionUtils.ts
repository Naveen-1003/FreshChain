/**
 * Transaction utilities for handling blockchain transaction hashes and explorer links
 */

export interface TransactionInfo {
  hash: string;
  isReal: boolean;
  explorerUrl?: string;
  displayText: string;
}

/**
 * Check if a transaction hash is real (from blockchain) or demo
 */
export function isRealTransactionHash(txHash: string): boolean {
  if (!txHash) return false;
  
  // Real transaction hashes are 66 characters (0x + 64 hex)
  if (txHash.length !== 66) return false;
  
  // Check if it starts with 0x and contains only hex characters
  const hexPattern = /^0x[a-fA-F0-9]{64}$/;
  return hexPattern.test(txHash);
}

/**
 * Check if a transaction hash is a demo hash
 */
export function isDemoTransactionHash(txHash: string): boolean {
  return txHash && txHash.startsWith('DEMO-');
}

/**
 * Get transaction information including explorer URL and display text
 */
export function getTransactionInfo(txHash: string): TransactionInfo {
  if (!txHash) {
    return {
      hash: '',
      isReal: false,
      displayText: 'N/A'
    };
  }

  if (isRealTransactionHash(txHash)) {
    return {
      hash: txHash,
      isReal: true,
      explorerUrl: `https://explorer-mezame.shardeum.org/tx/${txHash}`,
      displayText: `${txHash.slice(0, 6)}...${txHash.slice(-4)}`
    };
  }

  if (isDemoTransactionHash(txHash)) {
    return {
      hash: txHash,
      isReal: false,
      displayText: `Demo: ${txHash.slice(5, 13)}...`
    };
  }

  // Fallback for other formats
  return {
    hash: txHash,
    isReal: false,
    displayText: txHash.length > 10 ? `${txHash.slice(0, 8)}...` : txHash
  };
}

/**
 * Generate a transaction link component props
 */
export function getTransactionLinkProps(txHash: string) {
  const txInfo = getTransactionInfo(txHash);
  
  if (txInfo.isReal && txInfo.explorerUrl) {
    return {
      href: txInfo.explorerUrl,
      target: '_blank',
      rel: 'noopener noreferrer',
      className: 'text-blue-600 hover:text-blue-800 flex items-center space-x-1 cursor-pointer',
      title: `View transaction ${txInfo.hash} on Shardeum Explorer`
    };
  }

  return {
    className: 'text-gray-500 flex items-center space-x-1 cursor-not-allowed',
    title: 'Demo transaction - not available on blockchain explorer'
  };
}