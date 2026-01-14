using System;

namespace GameSecretaryApp;

public sealed class LaunchConfig
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
}
