plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.activeperception"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.activeperception"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    // QNN's libQnnHtp.so internally dlopen()s sibling Skel/Stub libs by path. With the
    // modern (default) compressed-in-APK packaging those paths don't exist on disk, so
    // dlopen returns "library not found". Force legacy extraction so the libs land in
    // /data/app/<id>/lib/arm64/ where dlopen can find them.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // On-device YOLOv8n. TFLite + Adreno GPU delegate is the working accelerator path on
    // S25 (Android 15+, Hexagon V79): ORT 1.22's QNN EP fails device-create on V79; TFLite
    // GPU delegate goes through OpenGL/Vulkan and is documented + maintained.
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.16.1")
    // Keep ORT around for OnnxYoloDetector fallback path (same Detector<Bitmap> interface).
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.22.0")

    // Cloud-offload HTTP client (OffloadClient) -- async POST of frames to the GPU server.
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}