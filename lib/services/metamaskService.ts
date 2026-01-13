declare global {
  interface Window {
    ethereum?: any;
  }
}

export interface MetaMaskAccount {
  address: string;
  balance?: string;
}

export interface TransactionRequest {
  to: string;
  data: string;
  value?: string;
  gas?: string;
  gasPrice?: string;
}

export class MetaMaskService {
  private ethereum: any;

  constructor() {
    if (typeof window !== 'undefined') {
      this.ethereum = window.ethereum;
    }
  }

  async isInstalled(): Promise<boolean> {
    return typeof this.ethereum !== 'undefined';
  }

  async isConnected(): Promise<boolean> {
    if (!this.ethereum) return false;
    
    try {
      const accounts = await this.ethereum.request({ method: 'eth_accounts' });
      return accounts && accounts.length > 0;
    } catch (error) {
      console.error('Error checking MetaMask connection:', error);
      return false;
    }
  }

  async connect(): Promise<MetaMaskAccount[]> {
    if (!this.ethereum) {
      throw new Error('MetaMask is not installed');
    }

    try {
      const accounts = await this.ethereum.request({
        method: 'eth_requestAccounts',
      });

      return accounts.map((address: string) => ({ address }));
    } catch (error: any) {
      if (error.code === 4001) {
        throw new Error('User rejected the connection request');
      }
      throw new Error(`Failed to connect to MetaMask: ${error.message}`);
    }
  }

  async getAccounts(): Promise<string[]> {
    if (!this.ethereum) return [];

    try {
      return await this.ethereum.request({ method: 'eth_accounts' });
    } catch (error) {
      console.error('Error getting accounts:', error);
      return [];
    }
  }

  async getCurrentAccount(): Promise<string | null> {
    const accounts = await this.getAccounts();
    return accounts.length > 0 ? accounts[0] : null;
  }

  async getBalance(address: string): Promise<string> {
    if (!this.ethereum) return '0';

    try {
      const balance = await this.ethereum.request({
        method: 'eth_getBalance',
        params: [address, 'latest'],
      });
      
      // Convert from wei to ether
      return (parseInt(balance, 16) / Math.pow(10, 18)).toFixed(4);
    } catch (error) {
      console.error('Error getting balance:', error);
      return '0';
    }
  }

  async getChainId(): Promise<string> {
    if (!this.ethereum) return '0x1';

    try {
      return await this.ethereum.request({ method: 'eth_chainId' });
    } catch (error) {
      console.error('Error getting chain ID:', error);
      return '0x1';
    }
  }

  async switchToShardeum(): Promise<boolean> {
    if (!this.ethereum) return false;

    const shardeumChainId = '0x1fb7'; // 8119 in hex

    try {
      await this.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: shardeumChainId }],
      });
      return true;
    } catch (switchError: any) {
      // Chain not added to MetaMask
      if (switchError.code === 4902) {
        try {
          await this.ethereum.request({
            method: 'wallet_addEthereumChain',
            params: [
              {
                chainId: shardeumChainId,
                chainName: 'Shardeum Testnet',
                nativeCurrency: {
                  name: 'Shardeum',
                  symbol: 'SHM',
                  decimals: 18,
                },
                rpcUrls: ['https://api-mezame.shardeum.org/'],
                blockExplorerUrls: ['https://explorer.shardeum.org/'],
              },
            ],
          });
          return true;
        } catch (addError) {
          console.error('Error adding Shardeum network:', addError);
          return false;
        }
      }
      console.error('Error switching to Shardeum:', switchError);
      return false;
    }
  }

  async signTransaction(transaction: TransactionRequest): Promise<string> {
    if (!this.ethereum) {
      throw new Error('MetaMask is not available');
    }

    try {
      const accounts = await this.getAccounts();
      if (accounts.length === 0) {
        throw new Error('No accounts connected');
      }

      const txHash = await this.ethereum.request({
        method: 'eth_sendTransaction',
        params: [
          {
            from: accounts[0],
            ...transaction,
          },
        ],
      });

      return txHash;
    } catch (error: any) {
      if (error.code === 4001) {
        throw new Error('User rejected the transaction');
      }
      throw new Error(`Transaction failed: ${error.message}`);
    }
  }

  async signMessage(message: string): Promise<string> {
    if (!this.ethereum) {
      throw new Error('MetaMask is not available');
    }

    try {
      const accounts = await this.getAccounts();
      if (accounts.length === 0) {
        throw new Error('No accounts connected');
      }

      const signature = await this.ethereum.request({
        method: 'personal_sign',
        params: [message, accounts[0]],
      });

      return signature;
    } catch (error: any) {
      if (error.code === 4001) {
        throw new Error('User rejected the signature request');
      }
      throw new Error(`Signing failed: ${error.message}`);
    }
  }

  onAccountsChanged(callback: (accounts: string[]) => void): void {
    if (this.ethereum) {
      this.ethereum.on('accountsChanged', callback);
    }
  }

  onChainChanged(callback: (chainId: string) => void): void {
    if (this.ethereum) {
      this.ethereum.on('chainChanged', callback);
    }
  }

  removeAllListeners(): void {
    if (this.ethereum) {
      this.ethereum.removeAllListeners('accountsChanged');
      this.ethereum.removeAllListeners('chainChanged');
    }
  }
}

// Singleton instance
let metamaskService: MetaMaskService | null = null;

export const getMetaMaskService = (): MetaMaskService => {
  if (!metamaskService) {
    metamaskService = new MetaMaskService();
  }
  return metamaskService;
};

export default MetaMaskService;