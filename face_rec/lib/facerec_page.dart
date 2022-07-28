// ignore_for_file: prefer_const_constructors
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:image/image.dart' as imgpack;
import 'dart:io';
import 'dart:convert';
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
      controller = CameraController(cameras![1], ResolutionPreset.max, enableAudio: false);
      controller!.initialize().then((_) {
        if (!mounted) {
          return;
        }
        setState(() {});
        autocam();
      });
    } else {
      Fluttertoast.showToast(msg: '找不到相机');
    }
  }

  void autocam() async {
    if (controller!.value.isInitialized) {
      //check if controller is initialized
      image = await controller!.takePicture(); //capture image
      File file = File(image!.path);
      // Read a jpeg image from file path
      imgpack.Image? resizedImage = imgpack.decodeImage(file.readAsBytesSync());
      // Resize the image
      resizedImage = imgpack.copyResize(resizedImage!, width: 250, height: 250);
      // Save the resize image
      file = file
        ..writeAsBytesSync(imgpack.encodeJpg(resizedImage, quality: 100));
      var request = http.MultipartRequest('post', Uri.parse(API.testimgUrl));
      request.files.add(http.MultipartFile.fromBytes(
          'test_images', File(file.path).readAsBytesSync(),
          filename: file.path));
      int? userPhone = loginUserPreference!.getInt("phone");
      request.fields.addAll({'phone': '$userPhone'});
      var streamedResponse = await request.send();
      if (streamedResponse.statusCode == 200) {
        var response = await streamedResponse.stream.bytesToString();
        final result = json.decode(response);
        showDialogFunction(result['result']);
      } else {
        setState(() {
          Fluttertoast.showToast(msg: '异常');
        });
      }
    }
  }

  void showDialogFunction(String str) async {
    if (!mounted) {
          return;
        }
    await showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Text("识别结果"),
          titleTextStyle: TextStyle(
              fontWeight: FontWeight.bold, color: Colors.black, fontSize: 20),
          actions: [
            ElevatedButton(
                onPressed: () {
                  Navigator.of(context).pop();
                  autocam();
                },
                child: Text("确定")),
          ],
          content: Text(str),
        );
      },
    );
  }

  fetchUser() async {
    loginUserPreference = await SharedPreferences.getInstance();
  }

  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed.
    controller?.dispose();
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
          padding: EdgeInsets.symmetric(horizontal: 0, vertical: 10),
          height: MediaQuery.of(context).size.height,
          width: MediaQuery.of(context).size.width,
          child: controller == null
              ? Center(child: Text("Loading Camera..."))
              : !controller!.value.isInitialized
                  ? Center(
                      child: CircularProgressIndicator(),
                    )
                  : CameraPreview(controller!)),
    );
  }

  // FloatingActionButton takepicBTN() {
  //   return FloatingActionButton(
  //     onPressed: () {
  //       if (controller != null) {
  //         autocam();
  //       }
  //     },
  //     child: const Icon(Icons.camera_alt),
  //   );
  // }
}
