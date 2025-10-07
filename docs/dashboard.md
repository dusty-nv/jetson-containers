# 🚀 Jetson Containers Build Dashboard

An interactive web dashboard for monitoring and analyzing Jetson Containers build results across multiple workflow runs.

## 📊 Features

### **Real-time Build Monitoring**
- ✅ **Live build status** with success/failure/timeout/OOM tracking
- 📈 **Success rate calculation** and trend analysis
- 🏃‍♂️ **Multi-platform support** (Orin, Thor, etc.)
- ⏱️ **Build duration tracking** and performance metrics

### **Interactive Analysis**
- 🔍 **Advanced search** across packages, tags, and failure points
- 🎛️ **Multi-dimensional filtering** by status, runner, duration
- 📋 **Sortable results** by package, status, duration, timestamp
- 📄 **Paginated display** for large result sets

### **Historical Timeline**
- 📅 **Last 10 workflow runs** with comparison capabilities
- 🔄 **Run switching** via dropdown and timeline interface
- 📊 **Cross-run analysis** and trend identification
- 🎯 **Detailed run metadata** (SHA, timestamp, statistics)

### **Professional Deployment**
- 🌐 **GitHub Pages hosting** with automatic deployment
- 📱 **Responsive design** for desktop and mobile
- 🎨 **Modern UI** with professional styling
- 🔗 **Direct links** to GitHub Actions logs

## 🔗 Access

**Live Dashboard**: https://nvidia-ai-iot.github.io/jetson-containers/

## 🎯 Quick Start

### **Viewing Results**
1. **Visit the dashboard URL** after any build matrix completion
2. **Browse current results** in the interactive table
3. **Use search/filters** to find specific packages or issues
4. **Click timeline items** to view historical runs

### **Understanding Status Indicators**
- ✅ **Success**: Package built successfully
- ❌ **Build Fail**: Compilation or build errors
- ⏰ **Timeout**: Build exceeded time limit
- 💥 **OOM Killed**: Out of memory during build

### **Navigation**
- **Search Box**: Find packages by name, tag, or failure point
- **Status Filter**: Show only specific build outcomes
- **Runner Filter**: Filter by build platform (Orin, Thor)
- **Sort Options**: Order by package, status, duration, or time
- **Timeline**: Click any run to view its results

## 🔧 Setup & Configuration

### **Automatic Operation**
The dashboard automatically triggers after every "Sweep – Build Matrix" workflow completion on the `dev` branch. No manual intervention required for basic functionality.

### **Enhanced Features (Historical Timeline)**

For full functionality including historical runs timeline, a Personal Access Token is required:

#### **1. Create Personal Access Token**
1. Go to [GitHub Settings → Personal Access Tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. **Name**: `Jetson Containers Dashboard API`
4. **Expiration**: 90 days or no expiration
5. **Scopes**: Select `repo` and `workflow`
6. **Generate and copy** the token (starts with `ghp_`)

#### **2. Add Repository Secret**
1. Go to [Repository Settings → Secrets](https://github.com/NVIDIA-AI-IOT/jetson-containers/settings/secrets/actions)
2. Click "New repository secret"
3. **Name**: `DASHBOARD_PAT` (exactly this name)
4. **Value**: Paste your token
5. Click "Add secret"

#### **3. Verify Setup**
After the next build matrix run, check the dashboard workflow logs for:
```
✅ DASHBOARD_PAT is accessible (length: 40)
✅ Success! Found 5+ workflow runs
📊 Timeline will show multiple runs
```

## 🏗️ Architecture

### **Workflow Trigger**
```yaml
on:
  workflow_run:
    workflows: ["Sweep – Build Matrix"]
    types: [completed]
    branches: [dev]
```

### **Data Flow**
1. **Build Matrix** completes → uploads `results.json` artifact
2. **Dashboard Workflow** triggers → downloads current results
3. **Historical Data** fetched via GitHub API (requires PAT)
4. **HTML Dashboard** generated with embedded JavaScript
5. **GitHub Pages** deployment with automatic hosting

### **File Structure**
```
GitHub Pages Root:
├── index.html          # Main dashboard (generated from dashboard.html)
├── results.json        # Current run data
├── runs/               # Historical run data
│   ├── results-{id}.json
│   └── ...
└── logs/               # Detailed build logs (future feature)
    ├── run-{id}/
    └── ...
```

## 🛠️ Technical Details

### **Frontend Technologies**
- **Pure HTML/CSS/JavaScript** (no frameworks)
- **Responsive design** with CSS Grid and Flexbox
- **Modern browser APIs** (fetch, async/await)
- **Progressive enhancement** (works without JavaScript)

### **Backend Integration**
- **GitHub Actions** workflow automation
- **GitHub API** for historical data access
- **GitHub Pages** for static site hosting
- **Artifact system** for data persistence

### **Performance Optimizations**
- **Client-side filtering** for instant search
- **Pagination** for large datasets
- **Lazy loading** of historical runs
- **Efficient DOM updates** with minimal reflows

## 🔍 Troubleshooting

### **Dashboard Not Updating**
- **Check workflow status**: Visit [Actions tab](https://github.com/NVIDIA-AI-IOT/jetson-containers/actions)
- **Verify trigger**: Dashboard only runs after build matrix completion
- **Branch context**: Ensure build matrix ran on `dev` branch

### **Historical Timeline Empty**
- **PAT Setup**: Historical runs require `DASHBOARD_PAT` secret
- **Token Permissions**: Verify token has `repo` and `workflow` scopes
- **API Limits**: Check for rate limiting in workflow logs

### **Missing Build Results**
- **Artifact Upload**: Verify build matrix uploaded `results.json`
- **Download Permissions**: Check cross-workflow artifact access
- **File Paths**: Ensure artifact structure matches expected format

### **Performance Issues**
- **Large Datasets**: Use search/filtering to reduce displayed results
- **Browser Cache**: Hard refresh (Ctrl+F5) to clear cached data
- **Network**: Check connection for API-heavy operations

## 📈 Metrics & Analytics

### **Success Rate Tracking**
- **Overall percentage** across all packages and platforms
- **Platform-specific rates** (Orin vs Thor performance)
- **Trend analysis** over time with historical comparison
- **Failure pattern identification** for common issues

### **Performance Monitoring**
- **Build duration analysis** with average and outlier detection
- **Resource utilization** tracking (timeouts, OOM events)
- **Platform comparison** for optimization opportunities
- **Package-specific metrics** for targeted improvements

### **Failure Analysis**
- **Categorized failure points** (compilation, dependencies, runtime)
- **Frequency analysis** of common failure patterns
- **Platform correlation** (failures specific to hardware)
- **Historical tracking** of resolved vs persistent issues

## 🚀 Future Enhancements

### **Planned Features**
- 📊 **Advanced Analytics**: Trend charts and statistical analysis
- 🔔 **Notification System**: Slack/email alerts for failures
- 📝 **Detailed Logs**: Full build log hosting and search
- 🤖 **AI Insights**: Automated failure pattern recognition

### **Integration Opportunities**
- 🔗 **JIRA Integration**: Automatic issue creation for failures
- 📧 **Email Reports**: Scheduled summary reports
- 📊 **Grafana Dashboards**: Advanced metrics visualization
- 🔍 **Log Aggregation**: Centralized log search and analysis

## 🤝 Contributing

### **Dashboard Improvements**
- **UI/UX Enhancements**: Submit PRs for interface improvements
- **Feature Requests**: Open issues for new functionality
- **Bug Reports**: Report issues with detailed reproduction steps
- **Documentation**: Help improve this guide and inline docs

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/NVIDIA-AI-IOT/jetson-containers.git
cd jetson-containers

# Edit dashboard workflow
vim .github/workflows/sweep-publish-dashboard.yml

# Test locally (requires build artifacts)
python -m http.server 8000
# Visit http://localhost:8000 to view dashboard
```

## 📞 Support

### **Getting Help**
- **GitHub Issues**: [Report bugs or request features](https://github.com/NVIDIA-AI-IOT/jetson-containers/issues)
- **Discussions**: [Community support and questions](https://github.com/NVIDIA-AI-IOT/jetson-containers/discussions)
- **Documentation**: This guide and inline workflow comments

### **Common Questions**

**Q: Why isn't my build showing up?**
A: Dashboard only shows builds from the `dev` branch. Ensure your build matrix ran on `dev`.

**Q: How do I access older builds?**
A: Historical timeline requires PAT setup. Follow the "Enhanced Features" section above.

**Q: Can I customize the dashboard?**
A: Yes! The dashboard is generated from the workflow. Submit PRs for improvements.

**Q: How often does it update?**
A: Automatically after every build matrix completion. No manual refresh needed.

---

*Generated by the Jetson Containers Build Dashboard System*
*Last Updated: 2025-09-24*
