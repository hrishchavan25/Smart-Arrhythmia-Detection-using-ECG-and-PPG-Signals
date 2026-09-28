
import { useState } from 'react';
import {
  Activity,
  HeartPulse,
  LayoutDashboard,
  BrainCircuit,
  Database,
  Settings,
  Bell,
  Upload,
  CircleCheck,
  AlertTriangle,
  Clock,
  Menu,
  X,
  ArrowUpRight,
  ArrowDownRight,
  ShieldCheck,
  FileHeart,
} from 'lucide-react';

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  AreaChart,
  Area,
} from 'recharts';

import './App.css';
import FileUpload from './components/FileUpload.jsx';
import ModelVisualizations from "./ModelVisualizations";


// --------------------------------------------------
// Illustrative ECG and PPG signal data
// Replace with real backend data later.
// --------------------------------------------------

const ecgData = Array.from({ length: 100 }, (_, i) => ({
  time: i,
  value:
    Math.sin(i * 0.35) * 0.08 +
    (i % 25 === 12 ? 1.5 : 0) +
    (i % 25 === 11 ? -0.25 : 0) +
    (i % 25 === 13 ? -0.4 : 0),
}));

const ppgData = Array.from({ length: 100 }, (_, i) => ({
  time: i,
  value:
    Math.max(0, Math.sin(i * 0.22)) * 0.9 +
    Math.sin(i * 0.08) * 0.08,
}));

const uncertaintyData = Array.from({ length: 20 }, (_, i) => ({
  time: i + 1,
  uncertainty: 0.12 + Math.abs(Math.sin(i * 0.6)) * 0.45,
  confidence: 0.88 - Math.abs(Math.sin(i * 0.6)) * 0.35,
}));

const detectionRecords = [
  {
    id: 'CS-1024',
    patient: 'Patient 001',
    rhythm: 'Normal',
    confidence: '97.8%',
    uncertainty: 'Low',
    time: '10:42 AM',
    status: 'Normal',
  },
  {
    id: 'CS-1023',
    patient: 'Patient 002',
    rhythm: 'PVC',
    confidence: '91.4%',
    uncertainty: 'Moderate',
    time: '10:36 AM',
    status: 'PVC',
  },
  {
    id: 'CS-1022',
    patient: 'Patient 003',
    rhythm: 'Normal',
    confidence: '96.2%',
    uncertainty: 'Low',
    time: '10:28 AM',
    status: 'Normal',
  },
  {
    id: 'CS-1021',
    patient: 'Patient 004',
    rhythm: 'PVC',
    confidence: '84.7%',
    uncertainty: 'High',
    time: '10:15 AM',
    status: 'PVC',
  },
];

// --------------------------------------------------
// Reusable statistic card
// --------------------------------------------------

function StatCard({ title, value, subtitle, icon: Icon, trend, color }) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        <div className="stat-icon" style={{ color }}>
          <Icon size={21} />
        </div>
        {trend && (
          <span className="stat-trend">
            <ArrowUpRight size={14} />
            {trend}
          </span>
        )}
      </div>

      <p className="stat-title">{title}</p>
      <h2 className="stat-value">{value}</h2>
      <p className="stat-subtitle">{subtitle}</p>
    </div>
  );
}

// --------------------------------------------------
// Signal chart card
// --------------------------------------------------

function SignalCard({ title, description, data, color, unit, isUploaded }) {
  return (
    <div className="panel signal-panel">
      <div className="panel-heading">
        <div>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>

        <span className="live-indicator">
          <span className="live-dot" />
          {isUploaded ? 'Uploaded' : 'Sample'}
        </span>
      </div>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid
              stroke="#E6EEF7"
              strokeDasharray="4 4"
              vertical={false}
            />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 11, fill: '#7890A8' }}
              tickLine={false}
              axisLine={false}
              label={{
                value: 'Time (samples)',
                position: 'insideBottom',
                offset: -5,
                fontSize: 11,
                fill: '#7890A8',
              }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: '#7890A8' }}
              tickLine={false}
              axisLine={false}
              width={40}
              label={{
                value: unit,
                angle: -90,
                position: 'insideLeft',
                fontSize: 11,
                fill: '#7890A8',
              }}
            />
            <Tooltip
              contentStyle={{
                borderRadius: 10,
                border: '1px solid #DCEBFA',
                fontSize: 12,
              }}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="signal-footer">
        <span>
          <span className="legend-dot" style={{ background: color }} />
          {title} waveform
        </span>
        <span>{isUploaded ? 'Uploaded CSV' : 'Illustrative data'}</span>
      </div>
    </div>
  );
}

// --------------------------------------------------
// Main application
// --------------------------------------------------

function App() {
  const [activePage, setActivePage] = useState('Dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [uploadedSignals, setUploadedSignals] = useState(null);

  const ecgChartData = uploadedSignals
    ? uploadedSignals.ecg.map((value, time) => ({ time, value }))
    : ecgData;
  const ppgChartData = uploadedSignals
    ? uploadedSignals.ppg.map((value, time) => ({ time, value }))
    : ppgData;

  const navigation = [
    { name: 'Dashboard', icon: LayoutDashboard },
    { name: 'Signal Analysis', icon: Activity },
    { name: 'Detection History', icon: FileHeart },
    { name: 'Model Overview', icon: BrainCircuit },
    { name: 'Dataset', icon: Database },
    { name: 'Settings', icon: Settings },
  ];

  return (
    <div className="app-shell">

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-icon">
            <HeartPulse size={25} />
          </div>

          <div>
            <h2>CardioSense</h2>
            <p>Intelligent Cardiac Monitoring</p>
          </div>

          <button
            className="mobile-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close menu"
          >
            <X size={20} />
          </button>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav className="navigation">
          {navigation.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.name}
                className={`nav-item ${
                  activePage === item.name ? 'nav-active' : ''
                }`}
                onClick={() => {
                  setActivePage(item.name);
                  setSidebarOpen(false);
                }}
              >
                <Icon size={19} />
                <span>{item.name}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-help">
            <div className="help-icon">
              <ShieldCheck size={20} />
            </div>
            <h4>Research Prototype</h4>
            <p>
              An uncertainty-aware ECG and PPG analysis system.
            </p>
            <span className="prototype-tag">Prototype v1.0</span>
          </div>

          <div className="sidebar-user">
            <div className="user-avatar">CS</div>
            <div>
              <strong>CardioSense</strong>
              <p>Research workspace</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">

        {/* Header */}
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="menu-button"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open menu"
            >
              <Menu size={22} />
            </button>

            <div>
              <p className="breadcrumb">Workspace / {activePage}</p>
              <h1>{activePage}</h1>
            </div>
          </div>

          <div className="topbar-actions">
            <span className="system-status">
              <span className="status-dot" />
              UI Preview
            </span>

            <button
              className="notification-button"
              aria-label="Notifications"
            >
              <Bell size={19} />
              <span className="notification-dot" />
            </button>

            <div className="topbar-avatar">CS</div>
          </div>
        </header>

        {/* Dashboard */}
        <div className="dashboard-content">

          <section className="welcome-section">
            <div>
              <h2>Welcome to CardioSense</h2>
              <p>
                Monitor cardiac signals and explore uncertainty-aware
                arrhythmia detection.
              </p>
            </div>

            <button
              className="primary-button"
              onClick={() => setActivePage('Signal Analysis')}
            >
              <Upload size={17} />
              Analyze Signals
            </button>
          </section>
          <FileUpload onFileSelected={setUploadedSignals} />

          <div className="prototype-notice">
            <AlertTriangle size={18} />
            <p>
              <strong>Prototype preview:</strong> All values and waveforms
              shown below are illustrative, not real patient measurements
              or clinical predictions.
            </p>
          </div>

          {/* Statistics */}
          <section className="stats-grid">
            <StatCard
              title="Signals Analyzed"
              value="1,284"
              subtitle="Illustrative total"
              icon={Activity}
              trend="12.5%"
              color="#2563EB"
            />

            <StatCard
              title="Normal Rhythm"
              value="1,106"
              subtitle="Illustrative detections"
              icon={CircleCheck}
              trend="8.2%"
              color="#16A34A"
            />

            <StatCard
              title="PVC Detections"
              value="178"
              subtitle="Illustrative detections"
              icon={HeartPulse}
              trend="4.3%"
              color="#DC3545"
            />

            <StatCard
              title="Model Confidence"
              value="94.6%"
              subtitle="Placeholder metric"
              icon={BrainCircuit}
              color="#7C3AED"
            />
          </section>

          {/* Signals */}
          <section className="section-heading">
            <div>
              <h2>Signal Monitoring</h2>
              <p>ECG and PPG waveform visualization</p>
            </div>

            <span className="section-badge">Sample signals</span>
          </section>

          <section className="signals-grid">
            <SignalCard
              title="ECG Signal"
              description="Electrocardiogram waveform"
              data={ecgChartData}
              color="#2563EB"
              unit="Amplitude"
              isUploaded={Boolean(uploadedSignals)}
            />

            <SignalCard
              title="PPG Signal"
              description="Photoplethysmography waveform"
              data={ppgChartData}
              color="#DC3545"
              unit="Amplitude"
              isUploaded={Boolean(uploadedSignals)}
            />
          </section>

          {/* Lower panels */}
          <section className="lower-grid">

            <div className="panel uncertainty-panel">
              <div className="panel-heading">
                <div>
                  <h3>Predictive Uncertainty</h3>
                  <p>Illustrative Gaussian Process output</p>
                </div>

                <span className="uncertainty-icon">
                  <BrainCircuit size={19} />
                </span>
              </div>

              <div className="uncertainty-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={uncertaintyData}>
                    <CartesianGrid
                      stroke="#E6EEF7"
                      strokeDasharray="4 4"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="time"
                      tick={{ fontSize: 11, fill: '#7890A8' }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      domain={[0, 1]}
                      tick={{ fontSize: 11, fill: '#7890A8' }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip />
                    <Area
                      type="monotone"
                      dataKey="uncertainty"
                      stroke="#8B5CF6"
                      fill="#DDD6FE"
                      fillOpacity={0.7}
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div className="uncertainty-legend">
                <span>
                  <span className="legend-dot purple-dot" />
                  Illustrative uncertainty
                </span>
                <span>0 = lower · 1 = higher</span>
              </div>

              <div className="method-note">
                <BrainCircuit size={17} />
                <p>
                  Gaussian Processes can estimate predictive uncertainty
                  alongside predictions. Actual uncertainty values will
                  be calculated by the trained model.
                </p>
              </div>
            </div>

            <div className="panel model-panel">
              <div className="panel-heading">
                <div>
                  <h3>Model Overview</h3>
                  <p>Proposed system architecture</p>
                </div>

                <span className="model-status">Research</span>
              </div>

              <div className="model-flow">
                <div className="flow-step">
                  <div className="flow-icon blue-flow">
                    <Activity size={19} />
                  </div>
                  <div>
                    <strong>Signal Input</strong>
                    <p>ECG + PPG data</p>
                  </div>
                  <CircleCheck size={17} className="flow-check" />
                </div>

                <div className="flow-connector" />

                <div className="flow-step">
                  <div className="flow-icon green-flow">
                    <Database size={19} />
                  </div>
                  <div>
                    <strong>Preprocessing</strong>
                    <p>Filtering and feature extraction</p>
                  </div>
                  <CircleCheck size={17} className="flow-check" />
                </div>

                <div className="flow-connector" />

                <div className="flow-step">
                  <div className="flow-icon purple-flow">
                    <BrainCircuit size={19} />
                  </div>
                  <div>
                    <strong>Gaussian Process</strong>
                    <p>Prediction and uncertainty estimation</p>
                  </div>
                  <Clock size={17} className="flow-pending" />
                </div>

                <div className="flow-connector" />

                <div className="flow-step">
                  <div className="flow-icon red-flow">
                    <HeartPulse size={19} />
                  </div>
                  <div>
                    <strong>Detection Output</strong>
                    <p>Normal / PVC classification</p>
                  </div>
                  <Clock size={17} className="flow-pending" />
                </div>
              </div>

              <div className="model-footer">
                <span>Architecture preview</span>
                <ArrowUpRight size={16} />
              </div>
            </div>
          </section>

          {/* Detection history */}
          <section className="panel history-panel">
            <div className="panel-heading history-heading">
              <div>
                <h3>Recent Detection History</h3>
                <p>Illustrative records for dashboard preview</p>
              </div>

              <button
                className="text-button"
                onClick={() => setActivePage('Detection History')}
              >
                View all
                <ArrowUpRight size={15} />
              </button>
            </div>

            <div className="table-wrapper">
              <table className="detection-table">
                <thead>
                  <tr>
                    <th>Record ID</th>
                    <th>Patient</th>
                    <th>Rhythm</th>
                    <th>Confidence</th>
                    <th>Uncertainty</th>
                    <th>Time</th>
                  </tr>
                </thead>

                <tbody>
                  {detectionRecords.map((record) => (
                    <tr key={record.id}>
                      <td className="record-id">{record.id}</td>
                      <td>{record.patient}</td>
                      <td>
                        <span
                          className={`rhythm-badge ${
                            record.status === 'PVC'
                              ? 'pvc-badge'
                              : 'normal-badge'
                          }`}
                        >
                          {record.rhythm}
                        </span>
                      </td>
                      <td>{record.confidence}</td>
                      <td>
                        <span
                          className={`uncertainty-badge ${
                            record.uncertainty === 'High'
                              ? 'high-uncertainty'
                              : record.uncertainty === 'Moderate'
                              ? 'moderate-uncertainty'
                              : 'low-uncertainty'
                          }`}
                        >
                          {record.uncertainty}
                        </span>
                      </td>
                      <td className="time-cell">{record.time}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="table-footer">
              <span>Showing 4 illustrative records</span>
              <span>Prototype data only</span>
            </div>
          </section>

          <footer className="dashboard-footer">
            <span>CardioSense · Research Prototype</span>
            <span>For research and educational use only</span>
          </footer>

          <ModelVisualizations />

        </div>
      </main>
    </div>
  );
}

export default App;