// ignore_for_file: prefer_const_constructors
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class FaceRecPage extends StatefulWidget {
  final CameraDescription camera;
  const FaceRecPage({required this.camera, Key? key}) : super(key: key);
  static String tag = 'facerec-page';

  @override
  State<FaceRecPage> createState() => _FaceRecPageState();
}

class _FaceRecPageState extends State<FaceRecPage> {
  late CameraController controller;
  // XFile? pictureFile;

  @override
  void initState() {
    super.initState();
    controller = CameraController(
      widget.camera,
      ResolutionPreset.max,
    );
    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      setState(() {});
    });
  }

  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed.
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.indigo,
        centerTitle: true,
        title: Text(
          "Face Rec",
          style: TextStyle(
            fontStyle: FontStyle.italic,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: Column(
        children: [
          if (!controller.value.isInitialized) ...[
            SizedBox(
              child: Center(
                child: CircularProgressIndicator(),
              ),
            )
          ] else ...[
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Center(
                child: SizedBox(
                  height: 400,
                  width: 400,
                  child: CameraPreview(controller),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: FloatingActionButton(
                onPressed: () async {
                  final image = await controller.takePicture();
                  setState(() {});
                },
                child: const Icon(Icons.camera_alt),
              ),
        ),
          ]
        ]
      ),
    );
  }
}
