/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.media.ImageReader.OnImageAvailableListener;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.SystemClock;
import android.os.Vibrator;
import android.util.Size;
import android.util.TypedValue;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Toast;

import androidx.core.content.ContextCompat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity {
    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged SSD model.
    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final boolean TF_OD_API_IS_QUANTIZED = true;
    private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    private static final String TF_OD_API_LABELS_FILE = "labelmap.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private Detector detector;

    //private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;
    private long timestamp = 0;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private BorderedText borderedText;


    //SafeCrossing
    private boolean have_pb = false;
    private boolean have_pc = false;
    private boolean is_detecting = false;
    final float match_scores = 0.5f;
    private float pb_degree_direction;
    private float pc_degree_direction;
    private int count_second = 200;
    private float start_direction;
    private boolean left_check = false;
    private boolean right_check = false;
    private float left_dir = 0;
    private float right_dir = 0;


    protected void decision_making() {

        Vibrator vibe = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        //initialize audio
        final MediaPlayer mp_has_pc = MediaPlayer.create(this, R.raw.has_pc);
        final MediaPlayer mp_has_pb = MediaPlayer.create(this, R.raw.has_pb);
        final MediaPlayer mp_no_pc = MediaPlayer.create(this, R.raw.no_pc);
        final MediaPlayer mp_move_left = MediaPlayer.create(this, R.raw.move_left);
        final MediaPlayer mp_move_right = MediaPlayer.create(this, R.raw.move_right);
        final MediaPlayer done_detection = MediaPlayer.create(this, R.raw.done_detection);

        have_pb = false;
        have_pc = false;

        if (start_direction > 270) {
            left_dir = start_direction - 90;
            right_dir = (start_direction - 360) + 90;
        } else if (start_direction < 90) {
            left_dir = (start_direction + 360) - 90;
            right_dir = start_direction + 90;
        } else {
            left_dir = start_direction - 90;
            right_dir = start_direction + 90;
        }

        //repeat for detect pb and pc
        detect_image();
        Timer timer = new Timer();
        Timer timer1 = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if (!is_detecting) {
                    return;
                }
                vibe.vibrate(100);
                List<Detector.Recognition> decision_result_1 = new ArrayList<>(detect_image());
                for (final Detector.Recognition results : decision_result_1) {
                    //if have pedestrian bridge
                    if (results.getTitle().contains("pedestrian bridge") && results.getConfidence() >= match_scores) {
                        have_pb = true;
                        pb_degree_direction = degree_direction;
                    }
                    //if have pedestrian crossing
                    if (results.getTitle().contains("pedestrian crossing") && results.getConfidence() >= match_scores) {
                        pc_degree_direction = degree_direction;
                        have_pc = true;
                    }
                }
            }
        }, 8000, 1000);

        //do detect on surrounding
        timer1.schedule(new TimerTask() {
            @Override
            public void run() {
                if (!is_detecting) {
                    return;
                }
                if (!left_check) {
                    mp_move_left.start();
                }
                if (degree_direction == left_dir || degree_direction == left_dir+1||degree_direction == left_dir-1) {
                    left_check = true;
                }
                if (left_check && !right_check) {
                    mp_move_right.start();
                }
                if (degree_direction == right_dir || degree_direction == right_dir+1||degree_direction == right_dir-1 ) {
                    right_check = true;
                    timer.cancel();
                    timer1.cancel();
                    done_detection.start();
                    while (done_detection.isPlaying()) {
                        //wait mp done playing
                    }
                    if (have_pb) {
                        mp_has_pb.start();
                        while (mp_has_pb.isPlaying()) {
                            //wait mp done playing
                        }
                        lead(true);
                    } else if (have_pc) {
                        mp_has_pc.start();
                        while (mp_has_pc.isPlaying()) {
                            //wait mp done playing
                        }
                        lead(false);
                    } else {
                        mp_no_pc.start();
                        while (mp_no_pc.isPlaying()) {
                            //wait mp done playing
                        }
                        detection_vehicle();
                    }
                }
            }
        }, 8000, 50);


        /*
        new java.util.Timer().schedule(
                new java.util.TimerTask() {
                    @Override
                    public void run() {
                        timer.cancel();
                        if (is_detecting) {
                            if (have_pb) {
                                mp_has_pb.start();
                                while (mp_has_pb.isPlaying()) {
                                    //wait mp done playing
                                }
                                lead(true);
                            } else if (have_pc) {
                                mp_has_pc.start();
                                while (mp_has_pc.isPlaying()) {
                                    //wait mp done playing
                                }
                                lead(false);

                            } else {
                                mp_no_pc.start();
                                while (mp_no_pc.isPlaying()) {
                                    //wait mp done playing
                                }
                                detection_vehicle();
                            }
                        } else {
                            return;
                        }
                    }
                }, 10000);
         */
        return;
    }


    //lead user to desired place
    private void lead(boolean is_pb) {

        if (!is_detecting) {
            return;
        }
        Vibrator vibe = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        final MediaPlayer mp_move_to_left = MediaPlayer.create(this, R.raw.move_to_left);
        final MediaPlayer mp_move_to_right = MediaPlayer.create(this, R.raw.move_to_right);
        final MediaPlayer mp_right_direction_pc = MediaPlayer.create(this, R.raw.right_direction_pc);
        final MediaPlayer mp_right_direction_pb = MediaPlayer.create(this, R.raw.right_direction_pb);

        //for long press and double click input
        long[] pattern = {0, 100, 500, 100, 500};

        float item_degree;

        if (is_pb) {
            item_degree = pb_degree_direction;
        } else {
            item_degree = pc_degree_direction;
        }

        do {
            if (!is_detecting) {
                return;
            }
            double direction_result = (item_degree - degree_direction + 180 + 720) % 360 - 180;

            if (direction_result < 0) {
                mp_move_to_left.start();
            } else if (direction_result > 0) {
                mp_move_to_right.start();
            }
        }while (item_degree != degree_direction);

        vibe.vibrate(pattern, -1);
        if (is_pb) {
            mp_right_direction_pb.start();
        }else{
            mp_right_direction_pc.start();
            while (mp_right_direction_pc.isPlaying()) {
                //wait mp done playing
            }
            detection_vehicle();
        }
    }


    //detecting on vehicle
    private void detection_vehicle() {

        Vibrator vibe = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        //initialize audio
        final MediaPlayer mp_detect_car = MediaPlayer.create(this, R.raw.detect_car);
        final MediaPlayer mp_detect_motor = MediaPlayer.create(this, R.raw.detect_motor);
        final MediaPlayer mp_able_crossing = MediaPlayer.create(this, R.raw.able_crossing);
        final MediaPlayer mp_move_direction_to_right = MediaPlayer.create(this, R.raw.move_direction_to_right);
        final MediaPlayer mp_move_direction_to_left = MediaPlayer.create(this, R.raw.move_direction_to_left);

        mp_move_direction_to_right.start();

        Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                vibe.vibrate(100);
                if (!is_detecting) {
                    timer.cancel();
                    return;
                }
                List<Detector.Recognition> decision_result_1 = new ArrayList<>(detect_image());
                for (final Detector.Recognition results : decision_result_1) {
                    //if have car
                    if (results.getTitle().contains("car") && results.getConfidence() >= match_scores) {
                        mp_detect_car.start();
                        count_second = count_second + 50;
                    } //if have motor
                    else if (results.getTitle().contains("motor") && results.getConfidence() >= match_scores) {
                        mp_detect_motor.start();
                        count_second = count_second + 50;
                    } else {
                        count_second--;
                    }
                }
                if (count_second == 100){
                    mp_move_direction_to_left.start();
                }
                if (count_second > 250) { //avoid the detection too long
                    count_second = 250;
                }
                if (count_second < 1) {//when no detected any traffic
                    timer.cancel();
                    mp_able_crossing.start();
                    while (mp_able_crossing.isPlaying()) {
                        //wait the mp end
                    }
                    start_crossing_road();
                }
            }
        }, 0, 1000);

    }


    private void start_crossing_road() {
        Timer timer = new Timer();

        //blinking flashlight
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                camera2Fragment.turnFlash(true);
            }
        }, 0, 250);
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                camera2Fragment.turnFlash(false);
            }
        }, 125, 250);

        //start a looping alert sound
        final MediaPlayer mp_crossing_alert = MediaPlayer.create(this, R.raw.crossing_alert_sound);
        mp_crossing_alert.setLooping(true);
        mp_crossing_alert.start();

        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if (!is_detecting) {
                    camera2Fragment.turnFlash(false);
                    mp_crossing_alert.stop();
                    timer.cancel();
                    return;
                }
            }
        }, 0, 100);
        return;
    }


    public List<Detector.Recognition> detect_image() {

        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");
        readyForNextImage();

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        List<Detector.Recognition> return_results = detector.recognizeImage(croppedBitmap);

        runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Detector.Recognition> results = detector.recognizeImage(croppedBitmap);

                        //lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Detector.Recognition> mappedRecognitions = new ArrayList<Detector.Recognition>();
                        for (final Detector.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {
                                canvas.drawRect(location, paint);
                                cropToFrameTransform.mapRect(location);
                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }
                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();
                        computingDetection = false;
                        /*runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });*/
                        return_results.addAll(results);
                    }

                });

        return return_results;
    }


    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        //initialize audio
        MediaPlayer mp_safecrossing_ready = MediaPlayer.create(this, R.raw.safecrossing_ready);
        mp_safecrossing_ready.start();
        //initialize audio
        final MediaPlayer mp_detect_start = MediaPlayer.create(this, R.raw.detection_start);
        final MediaPlayer mp_detect_stop = MediaPlayer.create(this, R.raw.detection_stop);

        //for long press and double click input
        Vibrator vibe = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        long[] pattern = {0, 100, 500, 100, 500};
        final Context context = this;
        final GestureDetector.SimpleOnGestureListener listener = new GestureDetector.SimpleOnGestureListener() {

            @Override
            public boolean onDoubleTap(MotionEvent e) {

                if (!is_detecting) {
                    Toast.makeText(context, "Detection Start", Toast.LENGTH_SHORT).show();
                    vibe.vibrate(pattern, -1);
                    start_direction = degree_direction;
                    mp_detect_start.start();

                    is_detecting = true;
                    decision_making();

                    return true;
                }
                return true;
            }

            @Override
            public void onLongPress(MotionEvent e) {
                if (is_detecting) {
                    is_detecting = false;
                    Toast.makeText(context, "Detection Stop", Toast.LENGTH_SHORT).show();
                    vibe.vibrate(500);
                    mp_detect_stop.start();
                }
            }
        };

        final GestureDetector gdetector = new GestureDetector(listener);

        gdetector.setOnDoubleTapListener(listener);
        gdetector.setIsLongpressEnabled(true);

        getWindow().getDecorView().setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent event) {
                return gdetector.onTouchEvent(event);
            }
        });


        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            this,
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing Detector!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }


    @Override
    protected void processImage() {
        //detection_start();
    }


    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(
                () -> {
                    try {
                        detector.setUseNNAPI(isChecked);
                    } catch (UnsupportedOperationException e) {
                        LOGGER.e(e, "Failed to set \"Use NNAPI\".");
                        runOnUiThread(
                                () -> {
                                    Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                                });
                    }
                });
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(100));
    }
}
