import { NextPage } from 'next';
import { getAuthService, UserRole } from '../lib/services/authService';
import { RoleGuard } from '../components/auth/RoleGuard';
import BatchActions from '../components/batch/BatchActions';
import AdminWallet from '../components/admin/AdminWallet';
import DashboardLayout from '../components/layout/DashboardLayout';
import { Package, Settings, Truck, Eye } from 'lucide-react';

interface ProfessionalPageProps {
  onLogout?: () => void;
}

const ProfessionalPage: NextPage<ProfessionalPageProps> = ({ onLogout }) => {
  const authService = getAuthService();
  const user = authService.getUser();

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Access Denied</h2>
          <p className="text-gray-600">Please log in to access the professional dashboard.</p>
        </div>
      </div>
    );
  }

  return (
    <DashboardLayout title="Professional Dashboard">
      <div className="max-w-7xl mx-auto">{/* Role-specific Welcome */}
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl p-8 mb-8">
          <div className="flex items-center space-x-4">
            <div className="w-16 h-16 bg-white bg-opacity-20 rounded-xl flex items-center justify-center">
              {user.role === UserRole.ADMIN && <Settings className="h-8 w-8" />}
              {user.role === UserRole.RETAILER && <Package className="h-8 w-8" />}
              {user.role === UserRole.TRANSPORTER && <Truck className="h-8 w-8" />}
            </div>
            <div>
              <h1 className="text-3xl font-bold mb-2">
                Welcome, {user.username}!
              </h1>
              <p className="text-blue-100 text-lg">
                {user.role === UserRole.ADMIN && 'Full system control and oversight'}
                {user.role === UserRole.RETAILER && 'Manage batches and inventory'}
                {user.role === UserRole.TRANSPORTER && 'Update transport stages'}
              </p>
            </div>
          </div>
        </div>

        {/* Admin Wallet Section */}
        <RoleGuard allowedRoles={[UserRole.ADMIN]}>
          <div className="card mb-8">
            <h2 className="text-xl font-bold text-gray-900 mb-4">System Status</h2>
            <AdminWallet />
          </div>
        </RoleGuard>

        {/* Batch Management */}
        <div className="card mb-8">
          <h2 className="text-xl font-bold text-gray-900 mb-4">
            {user.role === UserRole.ADMIN && 'Batch Administration'}
            {user.role === UserRole.RETAILER && 'Batch Management'}
            {user.role === UserRole.TRANSPORTER && 'Transport Updates'}
          </h2>
          <BatchActions />
        </div>

        {/* Role-specific Information */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Your Capabilities</h3>
            <div className="space-y-3">
              {user.role === UserRole.ADMIN && (
                <>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">Create and manage all batches</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">Update any batch stage</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">View system analytics</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">Backend handles all transactions</span>
                  </div>
                </>
              )}
              {user.role === UserRole.RETAILER && (
                <>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">Create new product batches</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">Update to retail stages</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">Mark products as selling</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">No wallet setup required</span>
                  </div>
                </>
              )}
              {user.role === UserRole.TRANSPORTER && (
                <>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">Update transport stages</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">Mark delivery completion</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">Report transport issues</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">Simple UI - no technical setup</span>
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">System Architecture</h3>
            <div className="space-y-3">
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span className="text-sm text-gray-700">Backend handles all operations</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span className="text-sm text-gray-700">Automatic transaction signing</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span className="text-sm text-gray-700">Real-time status updates</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span className="text-sm text-gray-700">Secure role-based access</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default ProfessionalPage;