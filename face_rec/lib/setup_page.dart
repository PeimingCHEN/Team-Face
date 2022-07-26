// ignore_for_file: prefer_const_constructors
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:io';

class SetUpPage extends StatefulWidget {
  const SetUpPage({Key? key}) : super(key: key);
  static String tag = 'setup-page';

  @override
  State<SetUpPage> createState() => _SetUpPageState();
}

class _SetUpPageState extends State<SetUpPage> {
  List<CameraDescription>? cameras; //list out the camera available
  CameraController? controller; //controller for camera
  XFile? image; //for captured image

  @override
  void initState() {
    loadCamera();
    super.initState();
  }

  loadCamera() async {
    cameras = await availableCameras();
    if (cameras != null) {
      controller = CameraController(cameras![1], ResolutionPreset.max);
      //cameras[0] = first camera, change to 1 to another camera

      controller!.initialize().then((_) {
        if (!mounted) {
          return;
        }
        setState(() {});
      });
    } else {
      print("NO any camera found");
    }
  }

  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed.
    controller!.dispose();
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
          child: Column(children: [
        Container(
          padding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
          height: 400,
          width: 400,
          child: controller == null
              ? Center(child: Text("Loading Camera..."))
              : !controller!.value.isInitialized
                  ? Center(
                      child: CircularProgressIndicator(),
                    )
                  : RotationTransition(
                      turns: AlwaysStoppedAnimation(90 / 360),
                      child: CameraPreview(controller!),
                    ),
        ),
        takepic(),
        Container(
          //show captured image
          padding: EdgeInsets.all(30),
          child: image == null
              ? Text("No image captured")
              : Image.file(
                  File(image!.path),
                  height: 400,
                ),
          //display captured image
        )
      ])),
    );
  }
  //     body: Container(
  //       // padding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
  //       // height: MediaQuery.of(context).size.height,
  //       // width: MediaQuery.of(context).size.width,
  //       child: ListView(children: [
  //         if (controller == null) ...[
  //           CircularProgressIndicator(),
  //         ] else ...[
  //           RotationTransition(
  //             turns: AlwaysStoppedAnimation(90/360),
  //             child: CameraPreview(controller!),
  //             ),
  //           takepic(),
  //           if (image != null)
  //             Image.file(File(image!.path))
  //         ]
  //       ]),
  //     ),
  //   );
  // }

  FloatingActionButton takepic() {
    return FloatingActionButton(
      onPressed: () async {
        try {
          if (controller != null) {
            //check if contrller is not null
            if (controller!.value.isInitialized) {
              //check if controller is initialized
              print('fuckkkkkk');
              image = await controller!.takePicture(); //capture image
              print('success');
              setState(() {});
            }
          }
        } catch (e) {
          print('iadusfghsudfgaiudfgboahidb'); //show error
        }
      },
      child: const Icon(Icons.camera_alt),
    );
  }
}
