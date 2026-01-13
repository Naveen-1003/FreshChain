import { NextPage } from 'next';
import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/router';
import Script from 'next/script';
import {
  ShoppingBag,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Plus,
  IndianRupee,
  CircleDollarSign,
  CreditCard,
  Package,
} from 'lucide-react';

declare global {
  interface Window {
    Razorpay: any;
  }
}
import DashboardLayout from '../components/layout/DashboardLayout';
import { getAuthService, UserRole } from '../lib/services/authService';
import {
  getBackendService,
  ParentOffer,
  ParentOfferPayload,
  MarketplaceRequest,
  RetailerBidPayload,
  PaymentOrderResult,
} from '../lib/services/backendService';

const MarketplacePage: NextPage = () => {
  const router = useRouter();
  const authService = getAuthService();
  const backendService = getBackendService();
  const user = authService.getUser();

  const [isLoading, setIsLoading] = useState(true);
  const [isActionLoading, setIsActionLoading] = useState(false);
  const [activeAction, setActiveAction] = useState<string>('');
  const [parentBatches, setParentBatches] = useState<ParentOffer[]>([]);
  const [requests, setRequests] = useState<MarketplaceRequest[]>([]);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [razorpayLoaded, setRazorpayLoaded] = useState(false);

  const [offerForm, setOfferForm] = useState({
    productType: '',
    unit: 'kg',
    basePrice: '',
    totalQuantity: '',
    currency: 'INR',
    notes: '',
  });

  const [bidForm, setBidForm] = useState({
    parentId: '',
    quantity: '',
    bidPrice: '',
  });

  const [paymentOrder, setPaymentOrder] = useState<PaymentOrderResult | null>(null);
  const [paymentForm, setPaymentForm] = useState({
    requestId: '',
    orderId: '',
    paymentId: '',
  });

  const [fulfillForm, setFulfillForm] = useState({
    requestId: '',
    childBatchId: '',
    productType: '',
  });

  const isProducerRole = useMemo(
    () => [UserRole.ADMIN, UserRole.AGGREGATOR, UserRole.PRODUCER].includes(user?.role as UserRole),
    [user?.role]
  );
  const isRetailerRole = user?.role === UserRole.RETAILER;
  const isTransporterRole = user?.role === UserRole.TRANSPORTER;

  const draftParents = useMemo(
    () => parentBatches.filter((parent) => parent.status !== 'published'),
    [parentBatches]
  );
  const publishedParents = useMemo(
    () => parentBatches.filter((parent) => parent.status === 'published'),
    [parentBatches]
  );
  const fulfilledRequests = useMemo(
    () => requests.filter((request) => request.status === 'fulfilled'),
    [requests]
  );
  const selectedFulfillRequest = useMemo(
    () => requests.find((request) => request.requestId === fulfillForm.requestId),
    [requests, fulfillForm.requestId]
  );

  useEffect(() => {
    if (!authService.isAuthenticated()) {
      router.push('/login');
      return;
    }
    void loadMarketplace();
  }, [router, isRetailerRole]);

  const loadMarketplace = async () => {
    setIsLoading(true);
    setError('');
    try {
      const [offersRes, requestsRes] = await Promise.all([
        backendService.listParentOffers(isRetailerRole ? 'published' : undefined),
        backendService.listMarketplaceRequests(),
      ]);

      if (offersRes.success && offersRes.data) {
        setParentBatches(offersRes.data);
      } else {
        setError(offersRes.error || 'Failed to load parent batches');
      }

      if (requestsRes.success && requestsRes.data) {
        setRequests(requestsRes.data);
      } else if (requestsRes.error) {
        setError((prev) => prev || requestsRes.error || 'Failed to load requests');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load marketplace');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateParentBatch = async () => {
    if (!offerForm.productType || !offerForm.unit || !offerForm.basePrice || !offerForm.totalQuantity) {
      setError('Fill in all offer fields before publishing.');
      return;
    }

    setIsActionLoading(true);
    setActiveAction('create-parent');
    setError('');
    setSuccess('');

    const payload: ParentOfferPayload = {
      productType: offerForm.productType.trim(),
      unit: offerForm.unit.trim(),
      basePrice: Number(offerForm.basePrice),
      totalQuantity: Number(offerForm.totalQuantity),
      currency: offerForm.currency.trim() || 'INR',
      metadata: offerForm.notes ? { notes: offerForm.notes.trim() } : undefined,
    };

    try {
      const response = await backendService.createParentOffer(payload);
      if (response.success) {
        const parentNumber = response.data?.parentBatchNumber
          ? ` (${response.data.parentBatchNumber})`
          : '';
        setSuccess(`Harvest batch recorded${parentNumber}. Publish it to expose bids.`);
        setOfferForm({ productType: '', unit: 'kg', basePrice: '', totalQuantity: '', currency: 'INR', notes: '' });
        await loadMarketplace();
      } else {
        setError(response.error || 'Failed to publish offer');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to publish offer');
    } finally {
      setIsActionLoading(false);
      setActiveAction('');
    }
  };

  const handlePublishParent = async (parentId: string) => {
    setIsActionLoading(true);
    setActiveAction(`publish-${parentId}`);
    setError('');
    setSuccess('');

    try {
      const response = await backendService.publishParentOffer(parentId);
      if (response.success && response.data) {
        setSuccess(`Parent batch ${response.data.parentBatchNumber} is live on the marketplace.`);
        await loadMarketplace();
      } else {
        setError(response.error || 'Failed to publish parent batch');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to publish parent batch');
    } finally {
      setIsActionLoading(false);
      setActiveAction('');
    }
  };

  const handleCreateBid = async () => {
    if (!bidForm.parentId || !bidForm.quantity || !bidForm.bidPrice) {
      setError('Select an offer and enter bid details.');
      return;
    }

    setIsActionLoading(true);
    setActiveAction('create-bid');
    setError('');
    setSuccess('');

    const payload: RetailerBidPayload = {
      parentId: bidForm.parentId,
      quantity: Number(bidForm.quantity),
      bidPrice: Number(bidForm.bidPrice),
    };

    try {
      const response = await backendService.createMarketplaceRequest(payload);
      if (response.success) {
        setSuccess('Bid submitted. Awaiting producer approval.');
        setBidForm({ parentId: '', quantity: '', bidPrice: '' });
        setPaymentOrder(null);
        await loadMarketplace();
      } else {
        setError(response.error || 'Failed to submit request');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit request');
    } finally {
      setIsActionLoading(false);
      setActiveAction('');
    }
  };

  const handleApproveRequest = async (requestId: string) => {
    setIsActionLoading(true);
    setActiveAction(`approve-${requestId}`);
    setError('');
    setSuccess('');

    try {
      const response = await backendService.approveMarketplaceRequest(requestId);
      if (response.success) {
        setSuccess('Bid approved. Retailer can now proceed with payment.');
        await loadMarketplace();
      } else {
        setError(response.error || 'Failed to approve request');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to approve request');
    } finally {
      setIsActionLoading(false);
      setActiveAction('');
    }
  };

  const handleRejectRequest = async (requestId: string) => {
    setIsActionLoading(true);
    setActiveAction(`reject-${requestId}`);
    setError('');
    setSuccess('');

    try {
      const response = await backendService.rejectMarketplaceRequest(requestId);
      if (response.success) {
        setSuccess('Bid rejected. Quantity returned to parent batch.');
        await loadMarketplace();
      } else {
        setError(response.error || 'Failed to reject request');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reject request');
    } finally {
      setIsActionLoading(false);
      setActiveAction('');
    }
  };

  const handleCreatePaymentOrder = async (requestId: string) => {
    if (!razorpayLoaded) {
      setError('Razorpay SDK not loaded. Please refresh the page.');
      return;
    }

    setIsActionLoading(true);
    setActiveAction(`order-${requestId}`);
    setError('');
    setSuccess('');

    try {
      const response = await backendService.createMarketplacePaymentOrder(requestId);
      if (response.success && response.data) {
        const { orderId, amount, currency, order } = response.data;
        
        // Open Razorpay checkout
        const options = {
          key: process.env.NEXT_PUBLIC_RAZORPAY_KEY_ID || 'rzp_test_YOUR_KEY',
          amount: amount,
          currency: currency,
          name: 'FreshChain Marketplace',
          description: `Advance payment for request ${requestId}`,
          order_id: orderId,
          handler: async (razorpayResponse: any) => {
            // Payment successful - confirm with backend
            setIsActionLoading(true);
            setActiveAction('confirm-payment');
            try {
              const confirmResponse = await backendService.confirmMarketplacePayment(requestId, {
                paymentId: razorpayResponse.razorpay_payment_id,
                orderId: razorpayResponse.razorpay_order_id,
              });

              if (confirmResponse.success) {
                setSuccess('Payment confirmed successfully! Producer will now fulfill your order.');
                setPaymentOrder(null);
                setPaymentForm({ requestId: '', orderId: '', paymentId: '' });
                await loadMarketplace();
              } else {
                setError(confirmResponse.error || 'Failed to confirm payment');
              }
            } catch (err) {
              setError(err instanceof Error ? err.message : 'Failed to confirm payment');
            } finally {
              setIsActionLoading(false);
              setActiveAction('');
            }
          },
          modal: {
            ondismiss: () => {
              setError('Payment cancelled. You can retry anytime.');
              setIsActionLoading(false);
              setActiveAction('');
            },
          },
          prefill: {
            name: user?.username || '',
            email: `${user?.username}@freshchain.local`,
          },
          theme: {
            color: '#14b8a6',
          },
        };

        const razorpay = new window.Razorpay(options);
        razorpay.open();
        setPaymentOrder(response.data);
      } else {
        setError(response.error || 'Unable to create payment order');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to create payment order');
    } finally {
      if (activeAction !== 'confirm-payment') {
        setIsActionLoading(false);
        setActiveAction('');
      }
    }
  };

  const handleConfirmPayment = async () => {
    if (!paymentForm.requestId || !paymentForm.orderId || !paymentForm.paymentId) {
      setError('Provide both Razorpay order ID and payment ID.');
      return;
    }

    setIsActionLoading(true);
    setActiveAction('confirm-payment');
    setError('');
    setSuccess('');

    try {
      const response = await backendService.confirmMarketplacePayment(paymentForm.requestId, {
        orderId: paymentForm.orderId,
        paymentId: paymentForm.paymentId,
      });

      if (response.success) {
        setSuccess('Payment confirmed. Await fulfillment from producer.');
        setPaymentOrder(null);
        setPaymentForm({ requestId: '', orderId: '', paymentId: '' });
        await loadMarketplace();
      } else {
        setError(response.error || 'Failed to confirm payment');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to confirm payment');
    } finally {
      setIsActionLoading(false);
      setActiveAction('');
    }
  };

  const handleFulfillRequest = async () => {
    if (!fulfillForm.requestId || !fulfillForm.childBatchId) {
      setError('Enter a child batch ID before fulfilling a request.');
      return;
    }

    setIsActionLoading(true);
    setActiveAction('fulfill-request');
    setError('');
    setSuccess('');

    try {
      const response = await backendService.fulfillMarketplaceRequest(fulfillForm.requestId, {
        childBatchId: fulfillForm.childBatchId.trim(),
        productType: fulfillForm.productType.trim() || undefined,
      });

      if (response.success) {
        setSuccess('Request fulfilled and child batch minted on-chain.');
        setFulfillForm({ requestId: '', childBatchId: '', productType: '' });
        await loadMarketplace();
      } else {
        setError(response.error || 'Failed to fulfill request');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fulfill request');
    } finally {
      setIsActionLoading(false);
      setActiveAction('');
    }
  };

  if (!user) {
    return null;
  }

  const canViewMarketplace = isProducerRole || isRetailerRole || isTransporterRole;

  if (!canViewMarketplace) {
    return (
      <DashboardLayout title="Marketplace">
        <div className="min-h-[60vh] flex items-center justify-center">
          <div className="text-center max-w-md">
            <AlertTriangle className="h-12 w-12 text-amber-500 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Marketplace Unavailable</h2>
            <p className="text-gray-600">
              Only producers, retailers, and transporters can access price discovery and payment workflows.
            </p>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout title="Marketplace">
      <Script
        src="https://checkout.razorpay.com/v1/checkout.js"
        onLoad={() => setRazorpayLoaded(true)}
        onError={() => setError('Failed to load Razorpay SDK')}
      />
      <div className="space-y-6 pb-12">
        <div className="glass-card-dark text-white rounded-lg p-4 flex items-center justify-between">
          <div>
            <p className="text-sm text-teal-300 font-medium">Bidding & Advance Settlement</p>
            <h1 className="text-2xl font-bold flex items-center space-x-2">
              <ShoppingBag className="h-6 w-6" />
              <span>FreshChain Marketplace</span>
            </h1>
            <p className="text-sm text-teal-200 mt-1">
              Farmers log parent batches → publish to marketplace • Retailers bid & pay advances • Transporters scan child batches and link sensors.
            </p>
          </div>
          <button
            onClick={loadMarketplace}
            className="bg-white bg-opacity-20 hover:bg-opacity-30 text-white px-3 py-2 rounded flex items-center space-x-2 text-sm"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>

        {(error || success) && (
          <div className="space-y-2">
            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-800 flex items-center space-x-2">
                <AlertTriangle className="h-4 w-4" />
                <span>{error}</span>
              </div>
            )}
            {success && (
              <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-lg text-emerald-800 flex items-center space-x-2">
                <CheckCircle className="h-4 w-4" />
                <span>{success}</span>
              </div>
            )}
          </div>
        )}

        {isProducerRole && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="glass-card p-4">
                <h2 className="text-lg font-semibold text-gray-900 mb-1 flex items-center space-x-2">
                  <Plus className="h-4 w-4 text-emerald-600" />
                  <span>Record Harvest Batch</span>
                </h2>
                <p className="text-xs text-gray-500 mb-4">
                  Store the full harvest quantity. FreshChain assigns the parent batch number automatically.
                </p>
                <div className="space-y-3">
                  <input
                    type="text"
                    placeholder="Product type"
                    value={offerForm.productType}
                    onChange={(e) => setOfferForm((prev) => ({ ...prev, productType: e.target.value }))}
                    className="input"
                  />
                  <div className="grid grid-cols-2 gap-3">
                    <input
                      type="text"
                      placeholder="Unit (kg, crate, etc)"
                      value={offerForm.unit}
                      onChange={(e) => setOfferForm((prev) => ({ ...prev, unit: e.target.value }))}
                      className="input"
                    />
                    <input
                      type="text"
                      placeholder="Currency"
                      value={offerForm.currency}
                      onChange={(e) => setOfferForm((prev) => ({ ...prev, currency: e.target.value }))}
                      className="input"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <input
                      type="number"
                      min="0"
                      step="0.01"
                      placeholder="Base price"
                      value={offerForm.basePrice}
                      onChange={(e) => setOfferForm((prev) => ({ ...prev, basePrice: e.target.value }))}
                      className="input"
                    />
                    <input
                      type="number"
                      min="0"
                      step="0.01"
                      placeholder="Total harvest quantity"
                      value={offerForm.totalQuantity}
                      onChange={(e) => setOfferForm((prev) => ({ ...prev, totalQuantity: e.target.value }))}
                      className="input"
                    />
                  </div>
                  <textarea
                    placeholder="Notes or grade details (optional)"
                    value={offerForm.notes}
                    onChange={(e) => setOfferForm((prev) => ({ ...prev, notes: e.target.value }))}
                    className="input"
                    rows={3}
                  />
                  <button
                    onClick={handleCreateParentBatch}
                    disabled={isActionLoading}
                    className="btn-primary w-full flex items-center justify-center space-x-2"
                  >
                    <Package className="h-4 w-4" />
                    <span>
                      {isActionLoading && activeAction === 'create-parent' ? 'Recording...' : 'Save Parent Batch'}
                    </span>
                  </button>
                  <p className="text-xs text-gray-500 text-center">
                    Parent batch numbers unlock once you publish them to the marketplace.
                  </p>
                </div>
              </div>

              <div className="glass-card p-4">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold text-gray-900">Draft Harvests</h2>
                  <span className="text-xs text-gray-500">{draftParents.length} waiting</span>
                </div>
                {draftParents.length === 0 ? (
                  <p className="text-sm text-gray-500">No drafts yet. Record a batch to prepare marketplace listings.</p>
                ) : (
                  <div className="space-y-3">
                    {draftParents.map((parent) => (
                      <div key={parent.parentId} className="border border-dashed border-amber-300 rounded-lg p-3">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-xs text-amber-600 font-semibold">{parent.parentBatchNumber}</p>
                            <h3 className="text-base font-semibold text-gray-900">{parent.productType}</h3>
                            <p className="text-sm text-gray-600">{parent.totalQuantity} {parent.unit} total harvest</p>
                          </div>
                          <div className="text-right">
                            <p className="text-xs text-gray-500">Base price</p>
                            <p className="text-lg font-bold text-emerald-600">{parent.basePrice} {parent.pricingCurrency}</p>
                          </div>
                        </div>
                        <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
                          <span>Recorded {new Date(parent.createdAt).toLocaleString()}</span>
                          <span>Available {parent.availableQuantity} {parent.unit}</span>
                        </div>
                        <button
                          onClick={() => handlePublishParent(parent.parentId)}
                          disabled={isActionLoading && activeAction === `publish-${parent.parentId}`}
                          className="btn-secondary text-xs mt-3"
                        >
                          {isActionLoading && activeAction === `publish-${parent.parentId}` ? 'Publishing...' : 'Publish to Marketplace'}
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="glass-card p-4">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold text-gray-900">Published Inventory</h2>
                  <span className="text-xs text-gray-500">Live: {publishedParents.length}</span>
                </div>
                {publishedParents.length === 0 ? (
                  <p className="text-sm text-gray-500">Publish a parent batch to allow retailers to place bids.</p>
                ) : (
                  <div className="space-y-3">
                    {publishedParents.map((parent) => (
                      <div key={parent.parentId} className="border border-emerald-200 rounded-lg p-3 bg-emerald-50 bg-opacity-30">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-xs text-emerald-600 font-semibold">{parent.parentBatchNumber}</p>
                            <h3 className="text-base font-semibold text-gray-900">{parent.productType}</h3>
                          </div>
                          <div className="text-right">
                            <p className="text-xs text-gray-500">Available</p>
                            <p className="text-lg font-bold text-emerald-700">{parent.availableQuantity} {parent.unit}</p>
                          </div>
                        </div>
                        <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-gray-600">
                          <span>Base • {parent.basePrice} {parent.pricingCurrency}</span>
                          <span>Published {parent.publishedAt ? new Date(parent.publishedAt).toLocaleString() : '—'}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                <p className="text-xs text-gray-500 mt-3">
                  Retailers can only bid on the parent batches listed above.
                </p>
              </div>
            </div>
          </div>
        )}

        {isRetailerRole && (
          <div className="glass-card p-4">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-1">
                <h2 className="text-lg font-semibold text-gray-900 mb-3">Submit Bid</h2>
                <div className="space-y-3">
                  <select
                    value={bidForm.parentId}
                    onChange={(e) => setBidForm((prev) => ({ ...prev, parentId: e.target.value }))}
                    className="input"
                  >
                    <option value="">Select producer offer</option>
                    {publishedParents.map((offer) => (
                      <option key={offer.parentId} value={offer.parentId}>
                        {offer.parentBatchNumber} • {offer.productType} • {offer.availableQuantity} {offer.unit}
                      </option>
                    ))}
                  </select>
                  <input
                    type="number"
                    min="0"
                    step="0.01"
                    placeholder="Quantity requested"
                    value={bidForm.quantity}
                    onChange={(e) => setBidForm((prev) => ({ ...prev, quantity: e.target.value }))}
                    className="input"
                  />
                  <input
                    type="number"
                    min="0"
                    step="0.01"
                    placeholder="Your price per unit"
                    value={bidForm.bidPrice}
                    onChange={(e) => setBidForm((prev) => ({ ...prev, bidPrice: e.target.value }))}
                    className="input"
                  />
                  <button
                    onClick={handleCreateBid}
                    disabled={isActionLoading}
                    className="btn-primary w-full flex items-center justify-center space-x-2"
                  >
                    <CircleDollarSign className="h-4 w-4" />
                    <span>
                      {isActionLoading && activeAction === 'create-bid' ? 'Submitting...' : 'Place Bid'}
                    </span>
                  </button>
                </div>
              </div>

              <div className="lg:col-span-2">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold text-gray-900">Your Requests</h2>
                  <span className="text-xs text-gray-500">Count: {requests.length}</span>
                </div>
                {requests.length === 0 ? (
                  <p className="text-sm text-gray-500">No requests submitted yet.</p>
                ) : (
                  <div className="space-y-3">
                    {requests.map((request) => (
                      <div key={request.requestId} className="border border-gray-200 rounded-lg p-3">
                        <div className="flex flex-wrap items-center justify-between">
                          <div>
                            <p className="text-xs text-gray-500">
                              {request.parentBatchNumber || request.parentId} • {request.parentProductType || 'Harvest'}
                            </p>
                            <h3 className="text-base font-semibold text-gray-900">
                              {request.quantity} units @ {request.bidPrice} {request.currency}
                            </h3>
                            {request.childBatchId && (
                              <p className="text-xs text-purple-600 mt-1">
                                Child batch minted: {request.childBatchId}
                              </p>
                            )}
                          </div>
                          <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                            request.status === 'pending_approval'
                              ? 'bg-amber-100 text-amber-700'
                              : request.status === 'approved'
                              ? 'bg-blue-100 text-blue-700'
                              : request.status === 'rejected'
                              ? 'bg-red-100 text-red-700'
                              : request.status === 'awaiting_payment'
                              ? 'bg-blue-100 text-blue-700'
                              : request.status === 'paid'
                              ? 'bg-emerald-100 text-emerald-700'
                              : request.status === 'fulfilled'
                              ? 'bg-purple-100 text-purple-700'
                              : 'bg-gray-100 text-gray-600'
                          }`}>
                            {request.status.toUpperCase().replace('_', ' ')}
                          </span>
                        </div>

                        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                          <div className="flex items-center space-x-2">
                            <IndianRupee className="h-4 w-4 text-gray-500" />
                            <span>
                              Advance {(request.advancePercent * 100).toFixed(0)}% ≈
                              {' '}
                              {(request.quantity * request.bidPrice * request.advancePercent).toFixed(2)} {request.currency}
                            </span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <CreditCard className="h-4 w-4 text-gray-500" />
                            <span>
                              {request.payment?.status === 'paid'
                                ? `Paid ${request.payment.amount / 100} ${request.payment.currency}`
                                : request.payment?.orderId
                                ? `Order: ${request.payment.orderId}`
                                : 'Payment pending'}
                            </span>
                          </div>
                        </div>

                        <div className="mt-3 flex flex-wrap gap-2">
                          {request.status === 'pending_approval' && (
                            <p className="text-xs text-amber-600">Awaiting producer approval...</p>
                          )}
                          {request.status === 'rejected' && (
                            <p className="text-xs text-red-600">This bid was rejected by the producer.</p>
                          )}
                          {request.status === 'approved' && (
                            <button
                              onClick={() => handleCreatePaymentOrder(request.requestId)}
                              className="btn-secondary text-xs"
                              disabled={isActionLoading || !razorpayLoaded}
                            >
                              {isActionLoading && activeAction === `order-${request.requestId}`
                                ? 'Opening Razorpay...'
                                : !razorpayLoaded
                                ? 'Loading payment...'
                                : 'Pay with Razorpay'}
                            </button>
                          )}
                          {request.status === 'awaiting_payment' && request.payment && (
                            <div className="text-xs text-blue-600">
                              <p>Payment pending for order {request.payment.orderId}</p>
                              <button
                                onClick={() => handleCreatePaymentOrder(request.requestId)}
                                className="btn-secondary text-xs mt-1"
                                disabled={isActionLoading || !razorpayLoaded}
                              >
                                Retry Payment
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {paymentForm.requestId && (
              <div className="mt-6 border-t border-gray-200 pt-4">
                <h3 className="text-base font-semibold text-gray-900 mb-2">Confirm Razorpay Payment</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <input
                    type="text"
                    value={paymentForm.orderId}
                    onChange={(e) => setPaymentForm((prev) => ({ ...prev, orderId: e.target.value }))}
                    placeholder="Order ID"
                    className="input"
                  />
                  <input
                    type="text"
                    value={paymentForm.paymentId}
                    onChange={(e) => setPaymentForm((prev) => ({ ...prev, paymentId: e.target.value }))}
                    placeholder="Payment ID"
                    className="input"
                  />
                  <button
                    onClick={handleConfirmPayment}
                    disabled={isActionLoading}
                    className="btn-primary flex items-center justify-center space-x-2"
                  >
                    <CreditCard className="h-4 w-4" />
                    <span>
                      {isActionLoading && activeAction === 'confirm-payment' ? 'Confirming...' : 'Confirm Payment'}
                    </span>
                  </button>
                </div>
                {paymentOrder && (
                  <p className="text-xs text-gray-500 mt-2">
                    Latest order ({paymentOrder.orderId}) value: {paymentOrder.amount / 100} {paymentOrder.currency}. Use Razorpay test keys to capture payment.
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {isProducerRole && (
          <div className="glass-card p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-semibold text-gray-900">Incoming Requests</h2>
              <span className="text-xs text-gray-500">{requests.length} total</span>
            </div>
            {requests.length === 0 ? (
              <p className="text-sm text-gray-500">No retailer bids yet.</p>
            ) : (
              <div className="space-y-3">
                {requests.map((request) => (
                  <div key={request.requestId} className="border border-gray-200 rounded-lg p-3">
                    <div className="flex flex-wrap items-center justify-between">
                      <div>
                        <p className="text-xs text-gray-500">
                          {request.parentBatchNumber || request.parentId} • Retailer: {request.retailer}
                        </p>
                        <h3 className="text-base font-semibold text-gray-900">
                          {request.quantity} @ {request.bidPrice} {request.currency}
                        </h3>
                      </div>
                      <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                        request.status === 'pending_approval'
                          ? 'bg-amber-100 text-amber-700'
                          : request.status === 'approved'
                          ? 'bg-blue-100 text-blue-700'
                          : request.status === 'rejected'
                          ? 'bg-red-100 text-red-700'
                          : request.status === 'paid'
                          ? 'bg-emerald-100 text-emerald-700'
                          : request.status === 'fulfilled'
                          ? 'bg-purple-100 text-purple-700'
                          : 'bg-gray-100 text-gray-700'
                      }`}>
                        {request.status.toUpperCase().replace('_', ' ')}
                      </span>
                    </div>
                    <div className="mt-2 text-sm text-gray-600">
                      <p>Requested on {new Date(request.createdAt).toLocaleString()}</p>
                      {request.approvedAt && (
                        <p className="text-xs text-blue-600">Approved {new Date(request.approvedAt).toLocaleString()}</p>
                      )}
                      {request.childBatchId && (
                        <p className="text-xs text-purple-600">Child batch: {request.childBatchId}</p>
                      )}
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {request.status === 'pending_approval' && (
                        <>
                          <button
                            onClick={() => handleApproveRequest(request.requestId)}
                            disabled={isActionLoading && activeAction === `approve-${request.requestId}`}
                            className="btn-primary text-xs"
                          >
                            {isActionLoading && activeAction === `approve-${request.requestId}` ? 'Approving...' : 'Approve Bid'}
                          </button>
                          <button
                            onClick={() => handleRejectRequest(request.requestId)}
                            disabled={isActionLoading && activeAction === `reject-${request.requestId}`}
                            className="btn-secondary text-xs bg-red-500 hover:bg-red-600 text-white"
                          >
                            {isActionLoading && activeAction === `reject-${request.requestId}` ? 'Rejecting...' : 'Reject Bid'}
                          </button>
                        </>
                      )}
                      {request.status === 'approved' && (
                        <p className="text-xs text-blue-600">Awaiting retailer payment...</p>
                      )}
                      {request.status === 'rejected' && (
                        <p className="text-xs text-red-600">This bid was rejected.</p>
                      )}
                    </div>
                    {request.status === 'paid' && (
                      <button
                        onClick={() => setFulfillForm({
                          requestId: request.requestId,
                          childBatchId: '',
                          productType: request.parentProductType || '',
                        })}
                        className="btn-primary text-xs mt-3"
                      >
                        Fulfill with Child Batch
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}

            {fulfillForm.requestId && (
              <div className="mt-6 border-t border-gray-200 pt-4">
                <h3 className="text-base font-semibold text-gray-900 mb-2">Mint Child Batch</h3>
                {selectedFulfillRequest && (
                  <p className="text-xs text-gray-500 mb-3">
                    Parent {selectedFulfillRequest.parentBatchNumber || selectedFulfillRequest.parentId} → Retailer {selectedFulfillRequest.retailer}
                  </p>
                )}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <input
                    type="text"
                    value={fulfillForm.childBatchId}
                    onChange={(e) => setFulfillForm((prev) => ({ ...prev, childBatchId: e.target.value }))}
                    placeholder="Child batch ID"
                    className="input"
                  />
                  <input
                    type="text"
                    value={fulfillForm.productType}
                    onChange={(e) => setFulfillForm((prev) => ({ ...prev, productType: e.target.value }))}
                    placeholder="Product type (optional)"
                    className="input"
                  />
                  <button
                    onClick={handleFulfillRequest}
                    disabled={isActionLoading}
                    className="btn-secondary flex items-center justify-center space-x-2"
                  >
                    <CheckCircle className="h-4 w-4" />
                    <span>
                      {isActionLoading && activeAction === 'fulfill-request' ? 'Fulfilling...' : 'Mark Fulfilled'}
                    </span>
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {isTransporterRole && (
          <div className="glass-card p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-semibold text-gray-900">Ready for Pickup</h2>
              <span className="text-xs text-gray-500">Fulfilled: {fulfilledRequests.length}</span>
            </div>
            {fulfilledRequests.length === 0 ? (
              <p className="text-sm text-gray-500">No paid marketplace batches have been minted yet. Check back after producers fulfill bids.</p>
            ) : (
              <div className="space-y-3">
                {fulfilledRequests.map((request) => (
                  <div key={request.requestId} className="border border-blue-200 rounded-lg p-3 bg-blue-50 bg-opacity-40">
                    <p className="text-xs text-blue-700 font-semibold">
                      {request.parentBatchNumber || request.parentId} • Retailer {request.retailer}
                    </p>
                    <h3 className="text-base font-semibold text-gray-900 mt-1">
                      Child batch {request.childBatchId || 'pending'}
                    </h3>
                    <p className="text-xs text-gray-600 mt-1">
                      Quantity {request.quantity} units
                      {request.parentProductType ? ` • ${request.parentProductType}` : ''}
                    </p>
                    <div className="mt-3 flex items-center justify-between">
                      <p className="text-xs text-gray-500">
                        Scan the QR from this child batch in the Consumer Audit screen to auto-link your transporter sensors.
                      </p>
                      {request.childBatchId && (
                        <button
                          onClick={() => router.push('/consumer-audit')}
                          className="btn-secondary text-xs"
                        >
                          Open QR Scanner
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {isLoading && (
          <div className="text-center py-8 text-sm text-gray-500">Loading marketplace data...</div>
        )}
      </div>
    </DashboardLayout>
  );
};

export default MarketplacePage;
