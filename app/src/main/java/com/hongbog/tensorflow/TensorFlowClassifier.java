package com.hongbog.tensorflow;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.support.v4.os.TraceCompat;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Vector;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_8UC1;

public class TensorFlowClassifier {
    private static final String TAG = "TFClassifier";
    private static final int MAX_RESULTS = 3;  // result 개수 제한
    private static final float THRESHOLD = 0.1f;  // outputs 값의 threshold 설정

    // neural network 관련 parameters
    private String[] inputNames;  // neural network 입력 노드 이름
    private String[] outputNames;  // neural network 출력 노드 이름
    private int[] imgSize = new int[2];
    private Vector<String> labels = new Vector<>();  // label 정보
    private int numClasses = 2;
    private int[] camSize = new int[]{7, 7};
    private float[] logits = new float[numClasses];  // logit 정보
    private float[] cam_outputs;
    private boolean runStats = false;

    private static final int WIDTH = 224;
    private static final int HEIGHT = 224;
    private float[] imgData = new float[WIDTH * HEIGHT * 3];

    private TensorFlowInferenceInterface tii;

    private TensorFlowClassifier() {

    }

    private static class SingleToneHolder {
        static final TensorFlowClassifier instance = new TensorFlowClassifier();
    }

    public static TensorFlowClassifier getInstance() {
        return SingleToneHolder.instance;
    }

    /**
     * 텐서플로우 classifier 생성 관련 초기화 함수
     * @param assetManager
     * @param modelFilename
     * @param labelFilename
     * @param width
     * @param height
     * @param inputNames
     * @param outputNames
     */
    public void createClassifier(
            AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            int width,
            int height,
            String[] inputNames,
            String[] outputNames) {
        this.inputNames = inputNames;
        this.outputNames = outputNames;
        this.imgSize[0] = width;
        this.imgSize[1] = height;
        this.cam_outputs = new float[6 * camSize[0] * camSize[1]];

        // label names 설정
        BufferedReader br = null;
        try {
            String actualFilename = labelFilename.split("file:///android_asset/")[1];
            br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
            String line = "";
            while((line = br.readLine()) != null) {
                this.labels.add(line);
            }
        } catch (IOException e) {
            Log.d(TensorFlowClassifier.TAG, e.toString());
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.d(TensorFlowClassifier.TAG, e.toString());
            }
        }

        this.tii = new TensorFlowInferenceInterface(assetManager, modelFilename);
    }


    /**
     * 치매 진단을 수행하는 함수
     * @param imgData
     * @return
     */
    public void dementiaDiagnosis(final float[] imgData) {
        long startTime = System.currentTimeMillis();

        TraceCompat.beginSection("dementiaDiagnosis");

        TraceCompat.beginSection("feed");
        tii.feed(this.inputNames[0], imgData, 1, this.imgSize[1], this.imgSize[0], 3);
        TraceCompat.endSection();

        TraceCompat.beginSection("run");
        tii.run(this.outputNames, this.runStats);
        TraceCompat.endSection();

        TraceCompat.beginSection("fetch");
        tii.fetch(this.outputNames[0], this.logits);
        tii.fetch(this.outputNames[1], this.cam_outputs);
        TraceCompat.endSection();
    }


    public String getClassificationLabel(){

        float maxValue = -1;
        int maxIndex = -1;

        for (int i = 0; i < this.numClasses; i++) {
            if (maxValue < this.logits[i]) {
                maxValue = this.logits[i];
                maxIndex = i;
            }
        }

        String result_txt;

        if (maxIndex == 0) {
            result_txt = "정상";
        } else {
            result_txt = "치매";
        }

        return result_txt;
    }


    public int getClassificationPercentage(){

        float maxValue = -1;

        for (int i = 0; i < this.numClasses; i++) {
            if (maxValue < this.logits[i]) {
                maxValue = this.logits[i];
            }
        }

        return Math.round(maxValue * 100);
    }


    private Bitmap createFixedScaleBitmap(final Bitmap bitmap){
        Bitmap bitmapTmp = Bitmap.createScaledBitmap(bitmap, WIDTH, HEIGHT, false);
        return bitmapTmp;
    }


    public float[] normalize(final Bitmap bitmap) {
        int mWidth = bitmap.getWidth();
        int mHeight = bitmap.getHeight();

        int[] ori_pixels = new int[mWidth * mHeight];
        float[] norm_pixels = new float[mWidth * mHeight * 3];

        bitmap.getPixels(ori_pixels, 0, mWidth, 0, 0, mWidth, mHeight);
        for (int i = 0; i < ori_pixels.length; i++) {
            int R = (ori_pixels[i] >> 16) & 0xff;
            int G = (ori_pixels[i] >> 8) & 0xff;
            int B = ori_pixels[i] & 0xff;

            norm_pixels[(i * 3) + 0] = (float) R / 255.0f;
            norm_pixels[(i * 3) + 1] = (float) G / 255.0f;
            norm_pixels[(i * 3) + 2] = (float) B / 255.0f;
        }
        return norm_pixels;
    }

    /**
     * 이 메소드 실행후
     * getClassificationLabel(), getClassificationPercentage() 을 실행 해야 한다.
     * @param bitmap
     */
    public void diagnosis(final Bitmap bitmap){
        Bitmap bitmapTmp = createFixedScaleBitmap(bitmap);
        imgData = normalize(bitmapTmp);
        dementiaDiagnosis(imgData);
    }


    /*private void gradcamVisualization() {
        Bitmap oriBitmap = ((BitmapDrawable) SelectActivity.imgOriginal.getDrawable()).getBitmap();
        oriBitmap = Bitmap.createScaledBitmap(oriBitmap, this.imgSize[0], this.imgSize[1], false);

        Size cam_size = new Size(this.camSize[0], this.camSize[1]);
        Size img_size = new Size(this.imgSize[0], this.imgSize[1]);

        // Original 이미지
        Mat oriMat = new Mat(img_size, CV_32F);
        Utils.bitmapToMat(oriBitmap, oriMat);

        // CAM 출력 값
        Mat camMat = new Mat(cam_size, CV_32F);
        camMat.put(0, 0, this.cam_outputs);
        Imgproc.resize(camMat, camMat, img_size);

        camMat.convertTo(camMat, CV_8UC1);
        Imgproc.applyColorMap(camMat, camMat, Imgproc.COLORMAP_JET);
        Imgproc.cvtColor(camMat, camMat,  Imgproc.COLOR_BGR2RGBA);

        camMat.convertTo(camMat, CV_32F, 0.35);
        Imgproc.accumulate(oriMat, camMat);

        camMat.convertTo(camMat, CV_8UC1);

        Utils.matToBitmap(camMat, oriBitmap);
        SelectActivity.imgCAM.setImageBitmap(oriBitmap);
    }*/


    public Bitmap gradcamVisualization(Bitmap oriBitmap) {
        oriBitmap = Bitmap.createScaledBitmap(oriBitmap, this.imgSize[0], this.imgSize[1], false);

        Size cam_size = new Size(this.camSize[0], this.camSize[1]);
        Size img_size = new Size(this.imgSize[0], this.imgSize[1]);

        // Original 이미지
        Mat oriMat = new Mat(img_size, CV_32F);
        Utils.bitmapToMat(oriBitmap, oriMat);

        // CAM 출력 값
        Mat camMat = new Mat(cam_size, CV_32F);
        camMat.put(0, 0, this.cam_outputs);
        Imgproc.resize(camMat, camMat, img_size);

        camMat.convertTo(camMat, CV_8UC1);
        Imgproc.applyColorMap(camMat, camMat, Imgproc.COLORMAP_JET);
        Imgproc.cvtColor(camMat, camMat,  Imgproc.COLOR_BGR2RGBA);

        camMat.convertTo(camMat, CV_32F, 0.35);
        Imgproc.accumulate(oriMat, camMat);

        camMat.convertTo(camMat, CV_8UC1);

        Utils.matToBitmap(camMat, oriBitmap);
        return oriBitmap;
    }
}
