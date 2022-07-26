// ignore_for_file: prefer_const_constructors
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:async';

class SetUpPage extends StatefulWidget {
  final CameraDescription camera;
  const SetUpPage({required this.camera, Key? key}) : super(key: key);
  static String tag = 'setup-page';

  @override
  State<SetUpPage> createState() => _SetUpPageState();
}

class _SetUpPageState extends State<SetUpPage> {
  late CameraController controller;
  // XFile? pictureFile;

  @override
  void initState() {
    super.initState();
    controller = CameraController(widget.camera, ResolutionPreset.medium,
        enableAudio: false);
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
          "Team Face",
          style: TextStyle(
            fontStyle: FontStyle.italic,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: Container(
        padding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
        height: MediaQuery.of(context).size.height,
        width: MediaQuery.of(context).size.width,
        child: ListView(children: [
          if (!controller.value.isInitialized) ...[
            CircularProgressIndicator(),
          ] else ...[
            CameraPreview(controller),
            takepic()
          ]
        ]),
      ),
    );
  }

  FloatingActionButton takepic() {
    return FloatingActionButton(
      onPressed: () async {
        pic();
      },
      child: const Icon(Icons.camera_alt),
    );
  }

  pic() {
    Timer(Duration(milliseconds: 3000), () {
      //after 3 seconds this will be called,
      final image = controller.takePicture();
      setState(() {});
    });
  }
}
