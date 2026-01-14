using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.Web.WebView2.Core;

namespace GameSecretaryApp;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Loaded += OnLoaded;
        Closing += OnClosing;
    }

    private async void OnLoaded(object sender, RoutedEventArgs e)
    {
        Title = "AI Game Secretary (Starting backend...)";

        var userData = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GameSecretaryApp",
            "WebView2"
        );
        Directory.CreateDirectory(userData);
        var env = await CoreWebView2Environment.CreateAsync(null, userData);
        await WebView.EnsureCoreWebView2Async(env);

        try
        {
            await BackendManager.Instance.StartAsync();
        }
        catch (Exception ex)
        {
            Title = "AI Game Secretary (Backend failed)";
            MessageBox.Show(ex.Message, "Backend start failed", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        Title = "AI Game Secretary";
        WebView.CoreWebView2.Navigate(BackendManager.Instance.DashboardUrl);
    }

    private void OnClosing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        try
        {
            BackendManager.Instance.Stop();
        }
        catch
        {
        }
    }
}
