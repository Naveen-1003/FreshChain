import React, { useState, useEffect } from 'react';
import { Wallet, AlertCircle, CheckCircle, RefreshCw, ExternalLink, Copy, Zap } from 'lucide-react';
import { getMetaMaskService, MetaMaskAccount } from '../../lib/services/metamaskService';
import { getBackendService } from '../../lib/services/backendService';

interface WalletStatus {
  connected: boolean;
  network: string;
  chainId: string;
  requiredChainId: string;
}

const AdminWallet: React.FC = () => {
  const [isConnecting, setIsConnecting] = useState(false);
  const [account, setAccount] = useState<MetaMaskAccount | null>(null);
  const [balance, setBalance] = useState<string>('0');
  const [chainId, setChainId] = useState<string>('');
  const [walletStatus, setWalletStatus] = useState<WalletStatus | null>(null);
  const [error, setError] = useState<string>('');
  const [isInstalled, setIsInstalled] = useState(false);

  const metamaskService = getMetaMaskService();
  const backendService = getBackendService();

  useEffect(() => {
    checkMetaMaskInstallation();
    loadWalletStatus();
    setupEventListeners();

    return () => {
      metamaskService.removeAllListeners();
    };
  }, []);

  const checkMetaMaskInstallation = async () => {
    const installed = await metamaskService.isInstalled();
    setIsInstalled(installed);
    
    if (installed) {
      const connected = await metamaskService.isConnected();
      if (connected) {
        await loadAccountInfo();
      }
    }
  };

  const loadWalletStatus = async () => {
    try {
      const status = await backendService.getWalletStatus();
      setWalletStatus(status);
    } catch (error) {
      console.error('Failed to load wallet status:', error);
    }
  };

  const loadAccountInfo = async () => {
    try {
      const currentAccount = await metamaskService.getCurrentAccount();
      if (currentAccount) {
        const balance = await metamaskService.getBalance(currentAccount);
        const chainId = await metamaskService.getChainId();
        
        setAccount({ address: currentAccount, balance });
        setBalance(balance);
        setChainId(chainId);
      }
    } catch (error) {
      console.error('Failed to load account info:', error);
    }
  };

  const setupEventListeners = () => {
    metamaskService.onAccountsChanged((accounts: string[]) => {
      if (accounts.length > 0) {
        loadAccountInfo();
      } else {
        setAccount(null);
        setBalance('0');
      }
    });

    metamaskService.onChainChanged((chainId: string) => {
      setChainId(chainId);
      loadAccountInfo();
    });
  };

  const connectWallet = async () => {
    if (!isInstalled) {
      window.open('https://metamask.io/download/', '_blank');
      return;
    }

    setIsConnecting(true);
    setError('');

    try {
      const accounts = await metamaskService.connect();
      if (accounts.length > 0) {
        await loadAccountInfo();
        
        // Check if we're on the right network
        const currentChainId = await metamaskService.getChainId();
        if (currentChainId !== '0x1fb7') {
          const switched = await metamaskService.switchToShardeum();
          if (switched) {
            await loadAccountInfo();
          }
        }
      }
    } catch (error: any) {
      setError(error.message);
    } finally {
      setIsConnecting(false);
    }
  };

  const switchNetwork = async () => {
    try {
      const switched = await metamaskService.switchToShardeum();
      if (switched) {
        await loadAccountInfo();
        setError('');
      }
    } catch (error: any) {
      setError(error.message);
    }
  };

  const copyAddress = () => {
    if (account?.address) {
      navigator.clipboard.writeText(account.address);
    }
  };

  const refreshBalance = async () => {
    if (account?.address) {
      await loadAccountInfo();
    }
  };

  const isCorrectNetwork = chainId === '0x1fb7';

  if (!isInstalled) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
        <div className="flex items-center space-x-3">
          <AlertCircle className="h-6 w-6 text-yellow-600" />
          <div>
            <h3 className="text-lg font-semibold text-yellow-800">MetaMask Required</h3>
            <p className="text-yellow-700 mt-1">
              MetaMask is required for admin operations. Please install it to continue.
            </p>
          </div>
        </div>
        <button
          onClick={connectWallet}
          className="mt-4 bg-yellow-600 hover:bg-yellow-700 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center space-x-2"
        >
          <ExternalLink className="h-4 w-4" />
          <span>Install MetaMask</span>
        </button>
      </div>
    );
  }

  if (!account) {
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Wallet className="h-6 w-6 text-blue-600" />
            <div>
              <h3 className="text-lg font-semibold text-blue-800">Connect MetaMask</h3>
              <p className="text-blue-700 mt-1">
                Connect your MetaMask wallet to perform admin operations
              </p>
            </div>
          </div>
          <button
            onClick={connectWallet}
            disabled={isConnecting}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center space-x-2"
          >
            {isConnecting ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <Wallet className="h-4 w-4" />
            )}
            <span>{isConnecting ? 'Connecting...' : 'Connect Wallet'}</span>
          </button>
        </div>
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Wallet Connection Status */}
      <div className={`border rounded-lg p-6 ${
        isCorrectNetwork ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {isCorrectNetwork ? (
              <CheckCircle className="h-6 w-6 text-green-600" />
            ) : (
              <AlertCircle className="h-6 w-6 text-yellow-600" />
            )}
            <div>
              <h3 className={`text-lg font-semibold ${
                isCorrectNetwork ? 'text-green-800' : 'text-yellow-800'
              }`}>
                {isCorrectNetwork ? 'Wallet Connected' : 'Wrong Network'}
              </h3>
              <p className={`mt-1 ${
                isCorrectNetwork ? 'text-green-700' : 'text-yellow-700'
              }`}>
                {isCorrectNetwork 
                  ? 'Connected to Shardeum Testnet' 
                  : 'Please switch to Shardeum Testnet'
                }
              </p>
            </div>
          </div>
          {!isCorrectNetwork && (
            <button
              onClick={switchNetwork}
              className="bg-yellow-600 hover:bg-yellow-700 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center space-x-2"
            >
              <Zap className="h-4 w-4" />
              <span>Switch Network</span>
            </button>
          )}
        </div>
      </div>

      {/* Account Information */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Account Information</h3>
        
        <div className="space-y-4">
          {/* Address */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Wallet Address
            </label>
            <div className="flex items-center space-x-2">
              <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 font-mono text-sm">
                {account.address}
              </div>
              <button
                onClick={copyAddress}
                className="p-2 text-gray-500 hover:text-gray-700 transition-colors"
                title="Copy address"
              >
                <Copy className="h-4 w-4" />
              </button>
              <a
                href={`https://explorer.shardeum.org/address/${account.address}`}
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 text-gray-500 hover:text-gray-700 transition-colors"
                title="View on explorer"
              >
                <ExternalLink className="h-4 w-4" />
              </a>
            </div>
          </div>

          {/* Balance */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Balance
            </label>
            <div className="flex items-center space-x-2">
              <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2">
                <span className="font-semibold">{balance} SHM</span>
              </div>
              <button
                onClick={refreshBalance}
                className="p-2 text-gray-500 hover:text-gray-700 transition-colors"
                title="Refresh balance"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* Network Info */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Network
            </label>
            <div className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-2">
              <div className="flex items-center justify-between">
                <span>Shardeum Testnet</span>
                <span className="text-sm text-gray-500">Chain ID: {parseInt(chainId, 16)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* System Status */}
      {walletStatus && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">System Status</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-700">Backend Connected</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-700">Blockchain Ready</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-700">MetaMask Connected</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-700">Admin Privileges</span>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-600" />
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminWallet;