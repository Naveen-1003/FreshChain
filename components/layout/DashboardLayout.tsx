import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { 
  Package, 
  Eye, 
  Settings, 
  LogOut, 
  User, 
  Truck, 
  Camera,
  BarChart3,
  Shield,
  Bell,
  Menu,
  X,
  Home,
  Activity,
  Wifi,
  CheckCircle,
  QrCode,
  RefreshCw,
  MapPin,
  Zap,
  ShoppingBag,
  Sparkles,
  Thermometer,
} from 'lucide-react';
import { getAuthService, UserRole } from '../../lib/services/authService';

interface DashboardLayoutProps {
  children: React.ReactNode;
  title?: string;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children, title = 'Dashboard' }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const router = useRouter();
  const authService = getAuthService();
  const user = authService.getUser();
  const isProducerRole = user?.role === UserRole.AGGREGATOR || user?.role === UserRole.PRODUCER;

  const handleLogout = () => {
    authService.logout();
    router.push('/');
  };

  if (!user) {
    return null;
  }

  const getRoleColor = (role: string) => {
    switch (role) {
      case UserRole.ADMIN: return 'bg-red-100 text-red-800';
      case UserRole.PRODUCER: return 'bg-orange-100 text-orange-800';
      case UserRole.AGGREGATOR: return 'bg-orange-100 text-orange-800';
      case UserRole.RETAILER: return 'bg-blue-100 text-blue-800';
      case UserRole.TRANSPORTER: return 'bg-green-100 text-green-800';
      case UserRole.CONSUMER: return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getRoleIcon = (role: string) => {
    switch (role) {
      case UserRole.ADMIN: return <Settings className="h-4 w-4" />;
      case UserRole.PRODUCER: return <Package className="h-4 w-4" />;
      case UserRole.AGGREGATOR: return <Package className="h-4 w-4" />;
      case UserRole.RETAILER: return <Package className="h-4 w-4" />;
      case UserRole.TRANSPORTER: return <Truck className="h-4 w-4" />;
      case UserRole.CONSUMER: return <Camera className="h-4 w-4" />;
      default: return <User className="h-4 w-4" />;
    }
  };

  const navigationItems = [
    // Admin Navigation
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: Home,
      roles: [UserRole.ADMIN],
      description: 'Overview & Analytics'
    },
    {
      name: 'QR Scanner',
      href: '/consumer-audit',
      icon: QrCode,
      roles: [UserRole.ADMIN],
      description: 'Verify Products'
    },
    {
      name: 'Freshness AI',
      href: '/freshness',
      icon: Sparkles,
      roles: [UserRole.ADMIN],
      description: 'AI Freshness Scanner'
    },
    
    // Aggregator Navigation (Creates Batches)
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: BarChart3,
      roles: [UserRole.AGGREGATOR, UserRole.PRODUCER],
      description: 'Batch Creation Hub'
    },
    {
      name: 'Freshness AI',
      href: '/freshness',
      icon: Sparkles,
      roles: [UserRole.AGGREGATOR, UserRole.PRODUCER],
      description: 'AI Freshness Scanner'
    },
    {
      name: 'Marketplace',
      href: '/marketplace',
      icon: ShoppingBag,
      roles: [UserRole.AGGREGATOR, UserRole.PRODUCER, UserRole.RETAILER],
      description: 'Pricing & Bids'
    },
    {
      name: 'QR Scanner',
      href: '/consumer-audit',
      icon: QrCode,
      roles: [UserRole.AGGREGATOR, UserRole.PRODUCER],
      description: 'Verify Products'
    },
    
    // Retailer Navigation
    {
      name: 'Freshness AI',
      href: '/freshness',
      icon: Sparkles,
      roles: [UserRole.RETAILER],
      description: 'AI Freshness Scanner'
    },
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: Package,
      roles: [UserRole.RETAILER],
      description: 'Batch Workflow View'
    },
    {
      name: 'QR Scanner',
      href: '/consumer-audit',
      icon: QrCode,
      roles: [UserRole.RETAILER],
      description: 'Product Verification'
    },
    
    // Transporter Navigation
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: Truck,
      roles: [UserRole.TRANSPORTER],
      description: 'Transport Management'
    },
    {
      name: 'QR Scanner',
      href: '/consumer-audit',
      icon: QrCode,
      roles: [UserRole.TRANSPORTER],
      description: 'Verify Batches'
    },
    
    // Consumer Navigation
    {
      name: 'QR Scanner',
      href: '/consumer-audit',
      icon: Camera,
      roles: [UserRole.CONSUMER],
      description: 'Scan Product QR Codes'
    },
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: Eye,
      roles: [UserRole.CONSUMER],
      description: 'Product Information'
    }
  ];

  const filteredNavigation = navigationItems.filter(item => 
    item.roles.includes(user.role as UserRole)
  );

  return (
    <div className="min-h-screen ocean-bg flex">
      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 transform ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      } transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0 lg:flex lg:flex-col`}>
        
        <div className="h-full sidebar-nav flex flex-col">
          {/* Logo & Role */}
          <div className="px-4 py-3 border-b border-teal-600 border-opacity-30 flex-shrink-0">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-8 h-8 bg-gradient-to-br from-teal-400 to-emerald-500 rounded-lg flex items-center justify-center shadow-lg">
                <Package className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-teal-900">FreshChain</h1>
                <p className="text-xs text-teal-700">Supply Chain Tracker</p>
              </div>
            </div>
            
            {/* User Role Badge */}
            <div className="flex items-center justify-between">
              <div className={`inline-flex items-center space-x-2 px-2 py-1 rounded text-sm font-medium ${getRoleColor(user.role)}`}>
                {getRoleIcon(user.role)}
                <span className="text-xs">{user.role}</span>
              </div>
              <button
                onClick={() => setSidebarOpen(false)}
                className="lg:hidden p-1 rounded-md text-teal-700 hover:text-teal-900"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* Navigation */}
          <nav className="px-3 py-2 space-y-1">
            {filteredNavigation.map((item) => {
              const isActive = router.pathname === item.href;
              return (
                <Link
                  key={`${item.name}-${item.href}`}
                  href={item.href}
                  className={`group flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                    isActive 
                      ? 'bg-teal-600 bg-opacity-70 text-white shadow-lg' 
                      : 'text-teal-800 hover:text-teal-900 hover:bg-teal-100 hover:bg-opacity-70'
                  }`}
                >
                  <item.icon className={`h-4 w-4 mr-2 ${isActive ? 'text-white' : 'text-teal-600'}`} />
                  <div className="flex-1">
                    <div className="font-medium text-sm">{item.name}</div>
                    <div className={`text-xs ${isActive ? 'text-teal-100' : 'text-teal-600'}`}>
                      {item.description}
                    </div>
                  </div>
                </Link>
              );
            })}
          </nav>

          {/* Quick Actions */}
          <div className="px-3 py-2 border-t border-teal-600 border-opacity-30">
            <div className="text-xs font-medium text-teal-700 mb-1">Quick Actions</div>
            <div className="space-y-1">
              {(isProducerRole || user.role === UserRole.RETAILER) && (
                <button
                  onClick={() => router.push('/marketplace')}
                  className="w-full flex items-center px-2 py-1.5 text-sm text-teal-700 hover:text-teal-900 hover:bg-teal-100 hover:bg-opacity-70 rounded-lg transition-colors"
                >
                  <ShoppingBag className="h-3 w-3 mr-2" />
                  <span className="text-xs">Marketplace</span>
                </button>
              )}
              <button
                onClick={() => window.location.reload()}
                className="w-full flex items-center px-2 py-1.5 text-sm text-teal-700 hover:text-teal-900 hover:bg-teal-100 hover:bg-opacity-70 rounded-lg transition-colors"
              >
                <RefreshCw className="h-3 w-3 mr-2" />
                <span className="text-xs">Refresh Data</span>
              </button>
            </div>
          </div>

          {/* Bottom Actions */}
          <div className="px-3 py-2 border-t border-teal-600 border-opacity-30">
            <button
              onClick={handleLogout}
              className="w-full flex items-center px-2 py-1.5 text-sm font-medium text-red-700 hover:text-red-900 hover:bg-red-100 hover:bg-opacity-70 rounded-lg transition-colors"
            >
              <LogOut className="h-3 w-3 mr-2" />
              <span className="text-xs">Logout</span>
            </button>
          </div>

          {/* Spacer to push everything to top */}
          <div className="flex-1"></div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Compact Top Header */}
        <div className="bg-white bg-opacity-95 backdrop-blur-sm border-b border-teal-200 border-opacity-30 px-6 py-3 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden p-2 rounded-md text-teal-600 hover:text-teal-800"
              >
                <Menu className="h-5 w-5" />
              </button>
              
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-teal-500 to-emerald-600 rounded-lg flex items-center justify-center">
                  <Package className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-teal-900">FreshChain</h1>
                  <p className="text-xs text-teal-600">Blockchain Supply Chain</p>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              {/* Network Status - Compact */}
              <div className="hidden md:flex items-center space-x-2">
                <div className="flex items-center space-x-1 px-2 py-1 bg-green-100 text-green-800 rounded text-xs">
                  <CheckCircle className="h-3 w-3" />
                  <span>Connected</span>
                </div>
                <div className="flex items-center space-x-1 px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                  <Zap className="h-3 w-3" />
                  <span>Shardeum</span>
                </div>
              </div>

              {/* User Actions */}
              <div className="flex items-center space-x-2">
                <button className="p-2 text-teal-600 hover:text-teal-800 relative">
                  <Bell className="h-4 w-4" />
                </button>
                <button
                  onClick={handleLogout}
                  className="text-xs text-teal-600 hover:text-teal-800 px-2 py-1 rounded hover:bg-teal-50 transition-colors"
                >
                  Logout
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Page Content */}
        <main className="flex-1 p-2 overflow-y-auto">
          {children}
        </main>

        {/* Compact Bottom Info */}
        <div className="bg-white bg-opacity-90 backdrop-blur-sm border-t border-teal-200 border-opacity-30 p-2 flex-shrink-0">
          <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              <div className="flex items-center space-x-2">
                <div className="w-5 h-5 bg-teal-100 rounded flex items-center justify-center">
                  <Package className="h-3 w-3 text-teal-600" />
                </div>
                <div>
                  <p className="text-xs font-medium text-teal-900">Frontend</p>
                  <p className="text-xs text-teal-600">Next.js • React • Tailwind</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <div className="w-5 h-5 bg-emerald-100 rounded flex items-center justify-center">
                  <Shield className="h-3 w-3 text-emerald-600" />
                </div>
                <div>
                  <p className="text-xs font-medium text-teal-900">Security</p>
                  <p className="text-xs text-teal-600">Role-based Access • Encrypted</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <div className="w-5 h-5 bg-blue-100 rounded flex items-center justify-center">
                  <Activity className="h-3 w-3 text-blue-600" />
                </div>
                <div>
                  <p className="text-xs font-medium text-teal-900">Blockchain</p>
                  <p className="text-xs text-teal-600">Shardeum • Immutable Audit Trail</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
};

export default DashboardLayout;