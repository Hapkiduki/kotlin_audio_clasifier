package com.hapkiduki.audioclasifier

import android.Manifest.permission.RECORD_AUDIO
import android.content.Context
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.annotation.RequiresApi
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import com.hapkiduki.audioclasifier.ui.theme.AudioClasifierTheme
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate

class MainActivity : ComponentActivity() {

    @RequiresApi(Build.VERSION_CODES.M)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val REQUEST_RECORD_AUDIO = 1337
        requestPermissions(arrayOf(RECORD_AUDIO), REQUEST_RECORD_AUDIO)

        setContent {
            AudioClasifierTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colors.background
                ) {
                    Greeting()
                }
            }
        }
    }
}

@Composable
fun Greeting() {
    val context = LocalContext.current


    var recorderSpecText by remember {
        mutableStateOf("")
    }

    var resultsText by remember {
        mutableStateOf("")
    }

    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(text = "Recording", modifier = Modifier.clickable {
            Toast.makeText(context, "Empieza a escuchar!", Toast.LENGTH_SHORT).show()
            loadAudio(context,
                onSpecChanged = {
                    recorderSpecText = it
                },
                onResultChanged = {
                    resultsText = it
                }
            )
        })
        if (recorderSpecText.isNotEmpty()) {
            Text(text = recorderSpecText)
        }
        if (resultsText.isNotEmpty()) {
            Text(text = resultsText)
        }
    }


}

fun loadAudio(
    context: Context,
    onSpecChanged: (String) -> Unit,
    onResultChanged: (String) -> Unit
) {

    val modelPath = "lite-model_yamnet_classification_tflite_1.tflite"
    val probabilityThreshold: Float = 0.3f

    val classifier = AudioClassifier.createFromFile(context, modelPath)
    val tensor = classifier.createInputTensorAudio()

    val format = classifier.requiredTensorAudioFormat
    val recorderSpecs = "Number Of Channels: ${format.channels}\n" +
            "Sample Rate: ${format.sampleRate}"

    onSpecChanged(recorderSpecs)


    // Empieza a grabar
    val record = classifier.createAudioRecord()
    record.startRecording()


    Timer().scheduleAtFixedRate(1, 500) {
        tensor.load(record)
        val output = classifier.classify(tensor)

        val filteredModelOutput = output[0].categories.filter {
            it.score > probabilityThreshold
        }

        val outputStr = filteredModelOutput.sortedBy { -it.score }
            .joinToString(separator = "\n") { "${it.label} -> ${it.score} " }

        onResultChanged(outputStr)
    }
}

@Preview(showBackground = true)
@Composable
fun DefaultPreview() {
    AudioClasifierTheme {
        Greeting()
    }
}