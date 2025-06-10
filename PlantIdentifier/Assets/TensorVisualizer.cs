using UnityEngine;
using Unity.InferenceEngine;
using System.IO;

public class TensorVisualizer : MonoBehaviour
{
    /// <summary>
    /// Converts a Tensor<float> in NCHW layout (1, C, H, W) to a Texture2D
    /// </summary>
    public Texture2D VisualizeTensorToTexture(Tensor<float> tensor)
    {
        int batch = tensor.shape[0]; // should be 1
        int channels = tensor.shape[1];
        int height = tensor.shape[2];
        int width = tensor.shape[3];

        Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float r = (channels > 0) ? tensor[0, 0, y, x] : 0f;
                float g = (channels > 1) ? tensor[0, 1, y, x] : 0f;
                float b = (channels > 2) ? tensor[0, 2, y, x] : 0f;

                tex.SetPixel(x, height - y - 1, new Color(r, g, b));
            }
        }

        tex.Apply();
        return tex;
    }

    public void SaveTexture(Texture2D tex, string path = "Assets/Output/input_preview.png")
    {
        string directory = Path.GetDirectoryName(path);
        if (!Directory.Exists(directory))
            Directory.CreateDirectory(directory); // Create folder if it doesn't exist

        byte[] bytes = tex.EncodeToPNG();
        File.WriteAllBytes(path, bytes);
        Debug.Log($"Saved tensor visualization to {path}");
    }
}
