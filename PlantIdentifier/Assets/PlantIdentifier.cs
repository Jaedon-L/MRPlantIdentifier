using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;

public class PlantIdentifier : MonoBehaviour
{
    [Header("Model & Labels")]
    public ModelAsset modelAsset;   // assign your best.onnx here
    public TextAsset labelsFile;   // one class name per line

    [Header("Test Input")]
    public Texture2D testImage;    // placeholder leaf image

    [Header("Settings")]
    [Range(0, 1)] public float confidenceThreshold = 0.25f;
    public int modelInputSize = 640;  // must match your ONNX export

    Worker engine;
    Tensor<float> inputTensor;
    string[] labels;
    public TensorVisualizer visualizer; // Drag this in from the scene


    public struct Detection
    {
        public Rect box;
        public string label;
        public float score;
    }

    void Start()
    {
        // 1) Load and compile the ONNX model (no extra graph ops needed here)
        var model = ModelLoader.Load(modelAsset);
        engine = new Worker(model, BackendType.GPUCompute);

        // 2) Prepare the input tensor shape [1,3,H,W]
        var shape = new TensorShape(1, 3, modelInputSize, modelInputSize);
        inputTensor = new Tensor<float>(shape);

        // 3) Load labels
        labels = labelsFile.text.Split('\n');
        labels = System.Array.FindAll(labels, l => !string.IsNullOrWhiteSpace(l));

        // 4) Run inference once on the placeholder
        var dets = Predict(testImage);
        foreach (var d in dets)
            Debug.Log($"Detected {d.label} @ {d.box} (conf {d.score:F2})");
    }

    public List<Detection> Predict(Texture2D src)
    {
        var padded = ScaleAndPad(src, modelInputSize);
        var pixels = padded.GetPixels();
        int pxCount = pixels.Length;

        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };

        for (int i = 0; i < pxCount; i++)
        {
            Color c = pixels[i];
            inputTensor[i + 0 * pxCount] = (c.r - mean[0]) / std[0];
            inputTensor[i + 1 * pxCount] = (c.g - mean[1]) / std[1];
            inputTensor[i + 2 * pxCount] = (c.b - mean[2]) / std[2];
        }

        Object.Destroy(padded);

        if (visualizer != null)
        {
            Texture2D preview = visualizer.VisualizeTensorToTexture(inputTensor);
            visualizer.SaveTexture(preview, "Assets/Output/model_input_debug.png");
        }

        engine.Schedule(inputTensor);

        var output = engine.PeekOutput(0) as Tensor<float>;
        var data = output.ReadbackAndClone();

        var dets = new List<Detection>();
        int rows = data.shape[1];
        int numClasses = labels.Length;

        for (int i = 0; i < rows; i++)
        {
            float objConf = 1f / (1f + Mathf.Exp(-data[0, i, 4]));
            if (objConf < confidenceThreshold)
                continue;

            // Find best class from logits
            int bestClass = -1;
            float bestLogit = float.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
            {
                float logit = data[0, i, 5 + c];
                if (logit > bestLogit)
                {
                    bestLogit = logit;
                    bestClass = c;
                }
            }

            if (bestClass < 0 || bestClass >= labels.Length)
                continue;

            // Optionally compute softmax + combined score
            float classProb = 0f;
            float expSum = 0f;
            for (int c = 0; c < numClasses; c++)
                expSum += Mathf.Exp(data[0, i, 5 + c] - bestLogit);
            classProb = Mathf.Exp(bestLogit - bestLogit) / expSum; // = 1 / expSum
            float finalScore = objConf * classProb;

            if (finalScore < confidenceThreshold)
                continue;

            // Bounding box
            float cx = data[0, i, 0], cy = data[0, i, 1];
            float w = data[0, i, 2], h = data[0, i, 3];

            float scaleX = (float)src.width / modelInputSize;
            float scaleY = (float)src.height / modelInputSize;

            float x = (cx - w / 2f) * scaleX;
            float y = (cy - h / 2f) * scaleY;
            float width = w * scaleX;
            float height = h * scaleY;

            if (width <= 0 || height <= 0 || width > src.width || height > src.height)
                continue;

            dets.Add(new Detection
            {
                box = new Rect(x, y, width, height),
                label = labels[bestClass],
                score = finalScore
            });
        }

        data.Dispose();
        output.Dispose();

        return ApplyNMS(dets, 0.45f);
    }

    Texture2D ScaleAndPad(Texture2D src, int targetSize)
    {
        float srcAR = (float)src.width / src.height;

        int resizedWidth, resizedHeight;
        if (srcAR > 1f)
        {
            resizedWidth = targetSize;
            resizedHeight = Mathf.RoundToInt(targetSize / srcAR);
        }
        else
        {
            resizedWidth = Mathf.RoundToInt(targetSize * srcAR);
            resizedHeight = targetSize;
        }

        // Step 1: Resize using RenderTexture and bilinear filter
        RenderTexture rt = RenderTexture.GetTemporary(resizedWidth, resizedHeight);
        RenderTexture.active = rt;
        Graphics.Blit(src, rt, new Vector2(1, 1), new Vector2(0, 0));

        Texture2D resized = new Texture2D(resizedWidth, resizedHeight, TextureFormat.RGB24, false);
        resized.ReadPixels(new Rect(0, 0, resizedWidth, resizedHeight), 0, 0);
        resized.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        FlipTextureVertically(resized);
        // Step 2: Create final padded texture
        Texture2D padded = new Texture2D(targetSize, targetSize, TextureFormat.RGB24, false);

        // Fill with gray
        Color fillColor = new Color(0.5f, 0.5f, 0.5f);
        Color[] grayPixels = new Color[targetSize * targetSize];
        for (int i = 0; i < grayPixels.Length; i++) grayPixels[i] = fillColor;
        padded.SetPixels(grayPixels);

        // Step 3: Copy resized image into the center of padded texture
        int xOffset = (targetSize - resizedWidth) / 2;
        int yOffset = (targetSize - resizedHeight) / 2;
        padded.SetPixels(xOffset, yOffset, resizedWidth, resizedHeight, resized.GetPixels());
        padded.Apply();


        return padded;
    }


    void OnDestroy()
    {
        inputTensor?.Dispose();
        engine?.Dispose();
    }
    void FlipTextureVertically(Texture2D tex)
    {
        int width = tex.width;
        int height = tex.height;
        Color[] pixels = tex.GetPixels();

        for (int y = 0; y < height / 2; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int top = y * width + x;
                int bottom = (height - y - 1) * width + x;

                // Swap pixels
                Color temp = pixels[top];
                pixels[top] = pixels[bottom];
                pixels[bottom] = temp;
            }
        }

        tex.SetPixels(pixels);
        tex.Apply();
    }
    List<Detection> ApplyNMS(List<Detection> dets, float iouThreshold = 0.5f)
    {
        var sorted = new List<Detection>(dets);
        sorted.Sort((a, b) => b.score.CompareTo(a.score));

        var results = new List<Detection>();

        while (sorted.Count > 0)
        {
            Detection best = sorted[0];
            results.Add(best);
            sorted.RemoveAt(0);

            sorted.RemoveAll(d =>
                d.label == best.label && ComputeIoU(best.box, d.box) > iouThreshold
            );
        }

        return results;
    }

    float ComputeIoU(Rect a, Rect b)
    {
        float x1 = Mathf.Max(a.xMin, b.xMin);
        float y1 = Mathf.Max(a.yMin, b.yMin);
        float x2 = Mathf.Min(a.xMax, b.xMax);
        float y2 = Mathf.Min(a.yMax, b.yMax);

        float interArea = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
        float unionArea = a.width * a.height + b.width * b.height - interArea;

        return interArea <= 0 ? 0 : interArea / unionArea;
    }


}
