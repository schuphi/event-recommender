# Free Deployment Guide 🚀

This guide covers deploying the Copenhagen Event Recommender system completely free using Railway, Vercel, GitHub Actions, and DuckDB.

## Overview

**Total Cost: $0/month** ✅

- **Frontend**: Vercel (via Lovable integration) - Free
- **Backend**: Railway free tier - Free  
- **Database**: DuckDB file - Free
- **Scrapers**: GitHub Actions - Free
- **Monitoring**: GitHub Actions - Free
- **Domain**: Railway/Vercel subdomain - Free

## 🚀 Automated Setup (What's Already Done)

The following has been automatically configured:

✅ **Railway Configuration** (`railway.toml`)  
✅ **Docker Setup** (Production-ready `Dockerfile`)  
✅ **GitHub Actions** for scrapers (`/.github/workflows/scrapers.yml`)  
✅ **GitHub Actions** for monitoring (`/.github/workflows/monitoring.yml`)  
✅ **Production Environment Variables** (`.env.example` updated)

## 📋 Manual Steps Required

### 1. Railway Backend Deployment

1. **Sign up at [Railway](https://railway.app/)**
   - Connect your GitHub account
   - Import this repository

2. **Configure Environment Variables in Railway**:
   ```
   ENVIRONMENT=production
   DATABASE_URL=/app/data/events.duckdb
   CORS_ORIGINS=https://your-frontend-domain.vercel.app
   PORT=8000
   ```

3. **Deploy**:
   - Railway will automatically detect the `Dockerfile`
   - Your API will be available at: `https://your-app.railway.app`

### 2. Frontend Deployment (Lovable + Vercel)

1. **Use Lovable to adjust frontend**:
   - Update API endpoint to your Railway URL
   - Adjust CORS settings
   - Deploy via Lovable's Vercel integration

2. **Note the frontend URL** for step 3

### 3. Update CORS Configuration

After getting your frontend URL:

1. **Update Railway Environment Variables**:
   ```
   CORS_ORIGINS=https://your-frontend-domain.vercel.app,https://your-app.railway.app
   ```

2. **Redeploy Railway app**

### 4. GitHub Actions Configuration

1. **Add Repository Secrets**:
   - Go to Settings → Secrets and Variables → Actions
   - Add these secrets:
     ```
     RAILWAY_API_URL=https://your-app.railway.app
     INSTAGRAM_USERNAME=your_instagram_username (optional)
     INSTAGRAM_PASSWORD=your_instagram_password (optional)
     ```

2. **Enable GitHub Actions**:
   - Go to Actions tab
   - Enable workflows if prompted

### 5. Database Setup

The system will automatically create a DuckDB database. To seed it:

1. **Run scrapers manually**:
   - Go to Actions → "Daily Scrapers" 
   - Click "Run workflow"

2. **Or run locally and upload**:
   ```bash
   python data-collection/scrapers/runner.py
   # Then copy data/events.duckdb to your Railway deployment
   ```

## 🔧 Configuration Details

### Railway Configuration

The `railway.toml` file configures:
- Dockerfile-based deployment
- Health check endpoint (`/health`)
- Environment variables
- Port configuration

### GitHub Actions

**Scrapers Workflow** (`.github/workflows/scrapers.yml`):
- Runs daily at 6 AM and 6 PM UTC
- Scrapes events from multiple sources
- Updates the database
- Can be triggered manually

**Monitoring Workflow** (`.github/workflows/monitoring.yml`):
- Health checks every 30 minutes
- Performance monitoring
- Automatic issue creation on failures
- Weekly reports

### Docker Configuration

The `Dockerfile`:
- Multi-stage build for optimization
- Production-ready Python setup
- Proper user permissions
- Health checks
- Environment variable handling

## 📊 Monitoring & Maintenance

### Automatic Health Checks
- API endpoint testing
- Database connectivity
- Performance monitoring
- Automatic issue creation

### Manual Monitoring
- Check Railway logs: Railway Dashboard → Deployments → Logs
- Monitor GitHub Actions: Repository → Actions tab
- API health: `https://your-app.railway.app/health`

### Database Maintenance
- GitHub Actions automatically run scrapers
- Database is file-based (DuckDB) - no maintenance needed
- Monitor event counts via `/stats` endpoint

## 🔍 Troubleshooting

### Common Issues

1. **API Not Responding**:
   - Check Railway deployment logs
   - Verify environment variables
   - Check Docker build logs

2. **CORS Errors**:
   - Verify CORS_ORIGINS includes your frontend URL
   - Check environment variable formatting

3. **Database Issues**:
   - DuckDB file missing: Will auto-create on first run
   - Permission issues: Check Docker user permissions

4. **Scrapers Failing**:
   - Check GitHub Actions logs
   - Verify API credentials in secrets
   - Rate limiting: Adjust scraper frequency

### Logs & Debugging

**Railway Logs**:
```bash
# In Railway dashboard
Deployments → Select deployment → View Logs
```

**GitHub Actions Logs**:
```bash
# In GitHub repository
Actions → Select workflow run → View logs
```

**API Health Check**:
```bash
curl https://your-app.railway.app/health
```

## 🎯 Performance & Limits

### Railway Free Tier Limits
- $5 monthly credit (usually sufficient)
- 500 hours execution time/month
- 1GB RAM, 1 vCPU
- 1GB storage

### GitHub Actions Limits
- 2,000 minutes/month (free)
- Current usage: ~30 minutes/month

### Optimization Tips
- Scrapers run twice daily (adjustable in workflow)
- Database cleanup can be added to workflows
- Monitor usage via Railway dashboard

## 🚀 Going Live Checklist

- [ ] Railway app deployed and responding
- [ ] Frontend deployed via Lovable/Vercel
- [ ] CORS configured correctly
- [ ] GitHub Actions secrets added
- [ ] Initial data scraping completed
- [ ] Health monitoring active
- [ ] API endpoints tested

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review Railway/GitHub logs
3. Verify all environment variables
4. Test individual components

**Ready to deploy!** 🎉

The system is now configured for completely free deployment with automatic data collection, monitoring, and health checks.