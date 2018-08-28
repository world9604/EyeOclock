package com.hongbog.view;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.hongbog.dementia.hongbogdementia.R;
import com.hongbog.tensorflow.TensorFlowClassifier;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class SelectActivity extends AppCompatActivity {
    private static final String TAG = "SelectActivity";
    private static final int PICTURE_REQUEST_CODE = 100;

    // Tensorflow parameter
    private static final String[] INPUT_NAMES = {"model/input_module/x"};
    private static final String[] OUTPUT_NAMES = {"model/output_module/softmax","grad_cam/outputs"};
    private static final int WIDTH = 224;
    private static final int HEIGHT = 224;
    private static final String MODEL_FILE = "file:///android_asset/model_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/label_strings.txt";

    public ImageView imgOriginal;
    public ImageView imgCAM;
    private TextView txtResult;

    private Button btnGallery;
    private Button btnDiagnosis;

    private Bitmap tmpBitmap;

    private Executor executor = Executors.newSingleThreadExecutor();

    private TensorFlowClassifier classifier = TensorFlowClassifier.getInstance();

    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("native-lib");
    }

    public static native void Divied(long matAddrInput, long matAddrResult);

    public static native void Nomal(long matAddrInput, long matAddrResult);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_select);

        imgOriginal = (ImageView) findViewById(R.id.imgOriginal);
        imgCAM = (ImageView) findViewById(R.id.imgCAM);
        txtResult = (TextView) findViewById(R.id.txtResult);

        btnGallery = (Button) findViewById(R.id.btnGallery);
        btnDiagnosis = (Button) findViewById(R.id.btnDiagnosis);

        btnGallery.setOnClickListener(new ButtonEventHandler());
        btnDiagnosis.setOnClickListener(new ButtonEventHandler());

        initTensorFlowAndLoadModel();
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try{
                    classifier.createClassifier(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            WIDTH,
                            HEIGHT,
                            INPUT_NAMES,
                            OUTPUT_NAMES);
                    Log.d(TAG, "Load Success");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    public class ButtonEventHandler implements View.OnClickListener {
        @Override
        public void onClick(View v) {
            int objectID = v.getId();

            if (objectID == R.id.btnGallery) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Audio.Media.EXTERNAL_CONTENT_URI);
                intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, false);
                intent.setType("image/*");
                startActivityForResult(Intent.createChooser(intent, "Select Image"), PICTURE_REQUEST_CODE);
            } else if (objectID == R.id.btnDiagnosis) {
                classifier.diagnosis(tmpBitmap);
                String label = classifier.getClassificationLabel();
                int percentage = classifier.getClassificationPercentage();
                Bitmap camBitmap = classifier.gradcamVisualization(tmpBitmap);
                imgCAM.setImageBitmap(camBitmap);
                txtResult.setText("Diagnostic results: " + label + " " + percentage + "%");
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICTURE_REQUEST_CODE) {
            if (resultCode == RESULT_OK) {
                imgOriginal.setImageResource(0);

                Uri uri = data.getData();

                if (uri != null) {
                    imgOriginal.setImageURI(uri);
                    Log.d("isroot--uri", String.valueOf(uri));
                    Bitmap bitmapOriginal = ((BitmapDrawable) imgOriginal.getDrawable()).getBitmap();
                    tmpBitmap = bitmapOriginal;
                }
            }
        }
    }
}