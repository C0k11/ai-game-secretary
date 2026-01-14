using System;
using System.IO;
using System.Text.Json;

namespace GameSecretaryApp;

public sealed class AppSettings
{
    public string LlmBaseUrl { get; set; } = "http://127.0.0.1:11434";
    public string LlmHost { get; set; } = "127.0.0.1";
    public int LlmPort { get; set; } = 11434;

    public bool StartLocalLlm { get; set; } = true;
    public bool AutoPullModel { get; set; } = false;
    public string OllamaModelsDir { get; set; } = @"D:\\Project\\ml_cache\\models";

    public string ModelName { get; set; } = "qwen2.5:14b-instruct";
    public string AdbSerial { get; set; } = "";
    public string[] OdQueries { get; set; } = Array.Empty<string>();
    public int Steps { get; set; } = 0;
    public bool NoLlm { get; set; } = false;
    public bool ForceRestart { get; set; } = false;

    private static string SettingsPath()
    {
        var dir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GameSecretaryApp"
        );
        Directory.CreateDirectory(dir);
        return Path.Combine(dir, "settings.json");
    }

    public static AppSettings Load()
    {
        try
        {
            var p = SettingsPath();
            if (!File.Exists(p)) return new AppSettings();
            var raw = File.ReadAllText(p);
            var s = JsonSerializer.Deserialize<AppSettings>(raw);
            return s ?? new AppSettings();
        }
        catch
        {
            return new AppSettings();
        }
    }

    public void Save()
    {
        var p = SettingsPath();
        var raw = JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(p, raw);
    }
}
