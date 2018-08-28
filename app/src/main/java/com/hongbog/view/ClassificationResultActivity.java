package com.hongbog.view;

import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.RelativeLayout;
import android.widget.TextView;

import com.hongbog.dementia.hongbogdementia.R;
import com.hongbog.tensorflow.TensorFlowClassifier;
import com.hongbog.util.ImageUtils;
import com.victor.loading.rotate.RotateLoading;

public class ClassificationResultActivity extends AppCompatActivity {

    private static final String TAG = ClassificationResultActivity.class.getSimpleName();
    private TensorFlowClassifier classifier;
    private TextView labelTextView;
    private TextView percentageTextView;
    private RelativeLayout loadingLayout;
    private RotateLoading rotateloading;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_classfication_result);

        initView();

        String bitmapPath = getIntent().getStringExtra(CaptureActivity.CAPTURE_PARCEL_BITMAP_EXTRA);

        new initTensorTask().execute(bitmapPath);
    }


    private void initView(){
        labelTextView = findViewById(R.id.label_textview);
        percentageTextView = findViewById(R.id.percentage_textview);
        loadingLayout = findViewById(R.id.loading_layout);
        rotateloading = findViewById(R.id.rotateloading);
    }


    private void startLoadingView() {
        rotateloading.start();
        loadingLayout.setVisibility(View.VISIBLE);
    }


    private void stopLoadingView() {
        loadingLayout.setVisibility(View.GONE);
        rotateloading.stop();
    }


    private class initTensorTask extends AsyncTask<String, Void, Bundle>{

        @Override
        protected void onPreExecute() {
            startLoadingView();
        }

        @Override
        protected Bundle doInBackground(String... bitmapPaths) {

            classifier = TensorFlowClassifier.getInstance();
            String bitmapPath = bitmapPaths[0];
            Bundle bundle = null;

            if(bitmapPath != null){
                Log.d(TAG, "bitmapPath : " + bitmapPath);
                Bitmap captureBitmap = ImageUtils.extractBitmapFromDirName(bitmapPath);

                if(captureBitmap == null) return null;

                classifier.diagnosis(captureBitmap);
                String label = classifier.getClassificationLabel();
                String percentage = String.valueOf(classifier.getClassificationPercentage());

                bundle = new Bundle();
                bundle.putString("label", label);
                bundle.putString("percentage", percentage);
            }

            return bundle;
        }

        @Override
        protected void onPostExecute(Bundle labelAndPercentageBundle) {
            if (labelAndPercentageBundle != null) {
                String label = labelAndPercentageBundle.getString("label");
                String percentage = labelAndPercentageBundle.getString("percentage");

                if(label == null || percentage == null) {
                    stopLoadingView();
                    return;
                }

                labelTextView.setText(label);
                percentageTextView.setText(percentage + "%");
            }
            stopLoadingView();
        }
    }
}