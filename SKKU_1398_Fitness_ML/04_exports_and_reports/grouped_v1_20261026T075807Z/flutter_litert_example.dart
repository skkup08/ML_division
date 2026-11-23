// Integration example only: NOT compiled or tested on Flutter/phone.
// Supply EXIF-corrected RGB, letterboxed to 640x640 (padding 114), /255,
// as planar CHW float32. See tensor_contract.json and skku_inference.py.
// Native LiteRT operator/runtime compatibility must be verified on target devices.
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
const equipmentNames = ['treadmill','seated_chest_press','shoulder_press','seated_leg_press','smith_machine'];
class EquipmentLiteRT {
  final Interpreter interpreter;
  EquipmentLiteRT(this.interpreter);
  static Future<EquipmentLiteRT> load() async => EquipmentLiteRT(
    await Interpreter.fromAsset('assets/skku1398_yolo11n_fp32.tflite'));
  List<Map<String,dynamic>> predictPrepared(Float32List chw, {
    required int originalWidth, required int originalHeight,
    required double scale, required int padLeft, required int padTop,
    required bool normalizedCoordinates, double confidence = .25,
  }) {
    if (interpreter.getInputTensor(0).shape.join(',') != '1,3,640,640' ||
        interpreter.getOutputTensor(0).shape.join(',') != '1,9,8400' ||
        chw.length != 3*640*640) {
      throw StateError('Model/input contract mismatch: inspect tensor_contract.json');
    }
    final input = [List.generate(3,(c)=>List.generate(640,(y)=>
      List.generate(640,(x)=>chw[c*640*640+y*640+x])))];
    final output = [List.generate(9,(_)=>List<double>.filled(8400,0))];
    interpreter.run(input,output);
    final candidates=<Map<String,dynamic>>[];
    for (var a=0;a<8400;a++) {
      var c=0;var score=output[0][4][a];
      for(var k=1;k<5;k++) {if(output[0][4+k][a]>score){score=output[0][4+k][a];c=k;}}
      if(score<confidence) continue;
      final unit=normalizedCoordinates?640.0:1.0;
      final cx=output[0][0][a]*unit,cy=output[0][1][a]*unit;
      final w=output[0][2][a]*unit,h=output[0][3][a]*unit;
      candidates.add({'class_id':c,'confidence':score,'box':<double>[cx-w/2,cy-h/2,cx+w/2,cy+h/2]});
    }
    candidates.sort((a,b)=>(b['confidence'] as double).compareTo(a['confidence'] as double));
    final kept=<Map<String,dynamic>>[];
    while(candidates.isNotEmpty && kept.length<300){
      final best=candidates.removeAt(0);kept.add(best);
      candidates.removeWhere((p)=>p['class_id']==best['class_id'] &&
        _iou(p['box'] as List<double>,best['box'] as List<double>)>.7);
    }
    return kept.map((p){
      final b=p['box'] as List<double>;final c=p['class_id'] as int;
      return <String,dynamic>{'class_id':c,'class_name':equipmentNames[c],
        'confidence':p['confidence'],'bbox_xyxy_pixels':[
          ((b[0]-padLeft)/scale).clamp(0.0,originalWidth.toDouble()),
          ((b[1]-padTop)/scale).clamp(0.0,originalHeight.toDouble()),
          ((b[2]-padLeft)/scale).clamp(0.0,originalWidth.toDouble()),
          ((b[3]-padTop)/scale).clamp(0.0,originalHeight.toDouble())]};
    }).toList();
  }
  void close()=>interpreter.close();
}
double _iou(List<double>a,List<double>b){
  final w=math.max(0.0,math.min(a[2],b[2])-math.max(a[0],b[0]));
  final h=math.max(0.0,math.min(a[3],b[3])-math.max(a[1],b[1]));
  final inter=w*h;
  return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-7);
}
