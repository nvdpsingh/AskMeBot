# GitHub Pages Setup Guide

## 🚀 Quick Setup Steps

### 1. Enable GitHub Pages
1. Go to your repository: `https://github.com/nvdpsingh/FastAPITut`
2. Click on **Settings** tab
3. Scroll down to **Pages** section in the left sidebar
4. Under **Source**, select **GitHub Actions**
5. Save the settings

### 2. Verify Deployment
1. Go to **Actions** tab in your repository
2. You should see the "Deploy AskMeBot to GitHub Pages" workflow
3. Click on it to see the deployment progress
4. Once completed, your site will be available at: `https://nvdpsingh.github.io/FastAPITut/`

### 3. Custom Domain (Optional)
If you want to use a custom domain:
1. In **Pages** settings, add your custom domain
2. Update the `CNAME` file in your repository
3. Configure DNS settings with your domain provider

## 🔧 Manual Deployment

If you need to deploy manually:

```bash
# Clone the repository
git clone https://github.com/nvdpsingh/FastAPITut.git
cd FastAPITut

# Run the deployment script
./deploy.sh local
```

## 📊 Monitoring

- **GitHub Actions**: Monitor build and deployment status
- **Pages**: Check site availability and performance
- **Issues**: Report any problems or bugs

## 🆘 Troubleshooting

### Common Issues:

1. **Workflow not running**: Check if GitHub Actions are enabled
2. **Build failing**: Check the Actions logs for specific errors
3. **Site not accessible**: Verify Pages source is set to GitHub Actions
4. **404 errors**: Ensure the workflow completed successfully

### Getting Help:
- Check the [GitHub Pages documentation](https://docs.github.com/en/pages)
- Review the [GitHub Actions logs](https://github.com/nvdpsingh/FastAPITut/actions)
- Open an [issue](https://github.com/nvdpsingh/FastAPITut/issues) for support

## 🎯 Next Steps

Once GitHub Pages is set up:
1. ✅ Your site will be live at `https://nvdpsingh.github.io/FastAPITut/`
2. ✅ Every push to `main` branch will trigger automatic deployment
3. ✅ You can monitor deployment status in the Actions tab
4. ✅ The site will automatically update with your latest changes

---

**Note**: The first deployment might take a few minutes. Subsequent deployments will be faster thanks to caching.
