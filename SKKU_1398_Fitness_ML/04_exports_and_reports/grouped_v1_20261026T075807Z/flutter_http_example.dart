// Example only: image → development API → JSON.
// This is NOT an on-device LiteRT integration.
// Flutter pubspec.yaml: dependencies: http: ^1.3.0 (verify current version)
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> recognizeEquipment({
  required File imageFile,
  required Uri endpoint, // e.g. Uri.parse('http://.../predict')
}) async {
  final req = http.MultipartRequest('POST', endpoint);
  req.files.add(await http.MultipartFile.fromPath('file', imageFile.path));
  final response = await http.Response.fromStream(await req.send());
  if (response.statusCode != 200) {
    throw Exception('Equipment inference failed: ${response.statusCode}');
  }
  return jsonDecode(response.body) as Map<String, dynamic>;
}
