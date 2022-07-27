// ignore_for_file: prefer_const_constructors
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:image/image.dart' as imgpack;
import 'dart:io';
import 'utils.dart';

class FaceRecPage extends StatefulWidget {
  const FaceRecPage({Key? key}) : super(key: key);
  static String tag = 'facerec-page';

  @override
  State<FaceRecPage> createState() => _FaceRecPageState();
}

class _FaceRecPageState extends State<FaceRecPage> {
  List<CameraDescription>? cameras; //list out the camera available
  CameraController? controller; //controller for camera
  XFile? image; //for captured image
  SharedPreferences? loginUserPreference; //get login user info

  @override
  void initState() {
    fetchUser();
    loadCamera();
    super.initState();
  }

  loadCamera() async {
    cameras = await availableCameras();
    if (cameras != null) {
      controller = CameraController(cameras![1], ResolutionPreset.max);
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

  fetchUser() async {
    loginUserPreference = await SharedPreferences.getInstance();
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
          // padding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
          height: 530,
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
        takepicBTN(),
      ])),
    );
  }

  FloatingActionButton takepicBTN() {
    return FloatingActionButton(
      onPressed: () async {
        try {
          if (controller != null) {
            //check if contrller is not null
            if (controller!.value.isInitialized) {
              //check if controller is initialized
              image = await controller!.takePicture(); //capture image
              File file = File(image!.path);
              // Read a jpeg image from file path
              imgpack.Image? resizedImage = imgpack.decodeImage(file.readAsBytesSync());
              // Resize the image
              resizedImage = imgpack.copyResize(resizedImage!, width: 400, height: 400);
              // Save the resize image
              file = file
                  ..writeAsBytesSync(imgpack.encodeJpg(resizedImage, quality: 100));
              int? userPhone = loginUserPreference!.getInt("phone");
              var request =
                  http.MultipartRequest('post', Uri.parse(API.testimgUrl));
              request.fields.addAll({'phone': '$userPhone'});
              request.files.add(http.MultipartFile.fromBytes(
                  'test_images', File(file.path).readAsBytesSync(),
                  filename: file.path));
              var res = await request.send();
              setState(() {});
            }
          }
        } catch (e) {
          print(e); //show error
        }
      },
      child: const Icon(Icons.camera_alt),
    );
  }
}
