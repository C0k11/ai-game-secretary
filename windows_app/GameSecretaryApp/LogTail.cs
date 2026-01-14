using System;
using System.IO;
using System.Text;

namespace GameSecretaryApp;

public static class LogTail
{
    public static string ReadTail(string path, int maxLines)
    {
        try
        {
            if (!File.Exists(path)) return "";
            var fi = new FileInfo(path);
            const int maxBytes = 256 * 1024;
            var take = (int)Math.Min(fi.Length, maxBytes);

            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            if (fs.Length > take)
            {
                fs.Seek(-take, SeekOrigin.End);
            }

            using var sr = new StreamReader(fs, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
            var text = sr.ReadToEnd();
            var lines = text.Replace("\r\n", "\n").Split('\n');
            if (lines.Length <= maxLines) return text;

            var sb = new StringBuilder();
            for (var i = Math.Max(0, lines.Length - maxLines); i < lines.Length; i++)
            {
                sb.AppendLine(lines[i]);
            }
            return sb.ToString();
        }
        catch
        {
            return "";
        }
    }
}
