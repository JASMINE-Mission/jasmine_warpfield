# 気になりメモ

* distortionの型制約がCallableなのは弱い気がするなぁ
* 「**」が厳密に何を指すかわかんないですが、引数で渡されたモノを書き換えるのは好ましくないかな？
* Detectorと、Opticsのcontains引数名は合わしておきたいかな
* withFITSIO -> WithFITSIOの方が良い
* FocalPlanePositionTableとDetectorPositionTableはassertionのためだけに存在してるのか...
  * どうするかなぁ。。。無くしたいなぁ...
  * 親クラスが子クラスを意識するのは避けたい。from_fitsfileはSourceTableにあるのがいいかも... 
* Legendreのプロパティに大文字のモノがある
* AltLegendreでcenterプロパティを多重定義してるの避けたい。
  * こういうケースってコンストラクタで値補正して親クラスのコンストラクタ呼び出すんじゃ無いかな？
* Legendre関係のクラス、多重継承しねいとイケないのかな？LegendreがBaseDestortion継承するのに実害あるのかな？
  * Sipも構造的には同じですね

```uml
@startuml
allowmixing 

package "astropy" {
  class SkyCoord
  class Angle
  class QTable
}

package "warpfield" {
  package "telescope" {
    Telescope -> SkyCoord
    Telescope -> Angle
    Telescope -> Optics
    Telescope *- Detector
    Optics -> SkyCoord
    Optics -> Angle
    Detector -> Angle
    Detector -> QTable
    package "distortion" {
      package "base.py" {
        class BaseDistortion {
          + __call__(position : np.ndarray) : ???
        }
      }
      package "sip.py" {
        AltSip -|> Sip
        SipDistortion  -|> Sip
        SipDistortion --|> BaseDistortion
        AltSipDistortion -|> AltSip
        AltSipDistortion --|> BaseDistortion
        class Sip <<dataclass>> {
          + order : int
          + center : np.ndarray
          + A : np.ndarray
          + B : np.ndarray
          + normalize(position : np.ndarray) : ???
          + apply(position : np.ndarray)
        }
        class AltSip <<dataclass>> {
          + center : np.ndarray
        }
        class SipDistortion
        class AltSipDistortion
      }
      package "legendre.py" {
        AltLegendre -|> Legendre
        LegendreDistortion  -|> Legendre
        LegendreDistortion --|> BaseDistortion
        AltLegendreDistortion -|> AltLegendre
        AltLegendreDistortion --|> BaseDistortion
        class Legendre <<dataclass>> {
          + order : int
          + center : np.ndarray
          + A : np.ndarray
          + B : np.ndarray
          + scale : float
          + normalize(position : np.ndarray) : ???
          + apply(position : np.ndarray)
        }
        class AltLegendre <<dataclass>> {
          + center : np.ndarray
        }
        class LegendreDistortion
        class AltLegendreDistortion
      }
    }
    package "source.py" {
      withFITSIO --> QTable
      withFITSIO ..> "generate" SourceTable
      SourceTable --|> withFITSIO
      SourceTable ---> SkyCoord
      FocalPlanePositionTable -|> SourceTable
      DetectorPositionTable -|> FocalPlanePositionTable
      Optics ---> SourceTable
      Optics ...> "generate" FocalPlanePositionTable
      Telescope ...> "generate" DetectorPositionTable
      Detector ...> "generate" DetectorPositionTable
      class withFITSIO <<dataclass>> {
        + table : QTable
        + {static} from_fitsfile(filename, key='table') : SourceTable
        + writeto(filename, overwrite=False)
      }
      class SourceTable <<dataclass>> {
        + skyCoord : SkyCoord
        + apply_space_motion(epoch)
      }
      class FocalPlanePositionTable <<dataclass>>
      class DetectorPositionTable <<dataclass>>
    }
    class Telescope <<dataclass>> {
      + pointing : SkyCoord
      + position_angle : Angle
      + optics : Optics
      + detectors : List[Detector]
      + get_footprints(frame, options) : List[SkyCoord?]
      + overlay_footprints(axis, options) :  axis?
      + display_focal_plane(axis, source, epoch, options)
      + observe(source, epoch, stack)
    }
    class Optics <<dataclass>> {
      + display_focal_plane : SkyCoord
      + position_angle : Angle
      + focal_length : Quantity
      + diameter : Quantity
      + field_of_view : Polygon
      + margin : Quantity
      + distortion : Callable
      + <<propeerty>> scale : ???
      + <<propeerty>> focal_plane_radius : ???
      + <<propeerty>> field_of_view_radius  : ???
      + <<propeerty>> projection  : ???
      + get_projection(frame) : ???
      + get_position_angle(frame) : Angle
      + get_fov_patch(options) : ???
      + set_distortion(distortion : Callable) 
      + contains(position) : boolean[]
      + imaging(sources, epoch)
    }
    class Detector <<detaclass>> {
      + naxis1 : int
      + naxis2 : int
      + pixel_scale : Quantity
      + offset_dx : Quantity
      + offset_dy : Quantity
      + position_angle : Angle
      + displacement : Callable
      + <<property>> width : ???
      + <<property>> height : ???
      + <<property>> xrange : ???
      + <<property>> yrange : ???
      + <<property>> corners : ???[][]
      + <<property>> detector_origin : ???[]
      + get_footprint_as_patch(options) : ???
      + get_first_line_as_patch(options) : ???
      + get_footprint_as_polygon(options) : Polygon
      + align(position) : ???
      + contains(pos) : boolean[]
      + capture(opsition)
    }
  }
}
@enduml
```
