# Sample: display on-sky footprints for dithering data

※"↑"は直前の呼び出しの戻り値（引数内ので関数コール）

※ boundaryは厳密にはbaundaryじゃないけど、関数群とクラスの見た目がわかるように便宜的に利用

```uml
@startuml

actor Client
participant "catalog:SourceTable" as catalog
boundary Visualize
boundary Util
boundary Jasmine
participant Optics
participant Detector
participant Telescope
boundary Source
participant "tmp:SourceTable" as tmp_SourceTable
box "astropy"
participant QTable
participant "pointing:QTable"
boundary "QTable.unique" as unique
participant SkyCoord 
participant WCS
end box
box "pandas"
participant "pandas.DataFrame" as pandas
end box
box "matplotlib"
participant pyplot
participant Figure
participant SubFigure
participant Polygon
end box

group 4
activate Client
Client -> QTable : read('./data/source_table.txt', format='ascii.ipac')
Client -> unique : unique(↑, 'catalog_id')
Client -> catalog : new(↑)
Client -> QTable : read('./data/pointing.txt', format='ascii.ipac')
QTable --> Client : return pointing
end
group 5 
Client -> "pointing:QTable" : to_pandas()
Client -> pandas : mean()
pandas --> Client : return center
Client -> SkyCoord : fov_center = SkyCoord (center.ra * u.deg, center.dec * u.deg).galactic
Client -> Visualize : fig, ax = get_subplot(fov_center)
  activate Visualize
  Visualize -> Util : get_projection(pointing key = 111, figsize=(8,8))
    activate  Util
    Util -> WCS : new (naxis=2)
    Util -> SkyCoord : いろいろ
    Util -> WCS : いろいろ
    Util --> Visualize : return WCS -> proj
    deactivate Util
  Visualize -> pyplot : fig = figure(figsize)
  Visualize -> Figure : axis = add_subplot(key, proj)
  Visualize --> Client : return fig, axis
  deactivate Visualize
Client -> Visualize : display_sources(axis, catalog, marker = '.', title='Gaia DR3 sources')
  activate Visualize
  Visualize ->   catalog : skycoord取得
  Visualize -> SubFigure : wcs.wcs.ctype
  Visualize -> Util : estimate_frame_from_ctype(ctype)
    activate Util
    Util --> Visualize : 'galactic' or 'icrs'
    deactivate Util
  Visualize -> catalog : lon, latの取得
  Visualize -> SubFigure : いろいろ
  deactivate Visualize
end
group 6
Client -> Visualize : fig, ax = get_subplot(fov_center)
note right : 既に書いているので省略
loop pointingの要素数分
Client -> Jasmine : get_jasmine(SkyCoord, Angle, Callable, bool)
  activate Jasmine
  Jasmine -> Optics : new
  Jasmine -> Detector : new 
  note right : 複数作ってるっぽい
  Jasmine -> Telescope : new 
  Jasmine --> Client : return Telescope
  deactivate Jasmine
end
Client -> Visualize  : display_sources(axis, catalog, marker = '.', title='Gaia DR3 sources') 
note right : 既に書いているので省略
loop 生成したTelescope分
Client -> Telescope : overlay_footprints(ax, , color=f'C{n+1}', lw=2)
  activate Telescope
  Telescope -> Util : estimate_frame_from_ctype(axis.wcs.wcs.ctype)
  note right : 既に書いているので省略
  Telescope -> Telescope : get_footprints(frame, **options)
    activate Telescope
    Telescope -> Optics : field_of_view
    Telescope -> Optics : get_projection(frame)
      activate Optics
      Optics -> SkyCoord : transform_to(frame)
      Optics -> Optics : get_position_angle(frame)
        activate Optics
        Optics -> SkyCoord : transform_to(frame)
        Optics -> SkyCoord : directional_offset_by(0.0, 1 * u.arcsec)
        Optics -> SkyCoord : position_angle(２つ上の呼び出しの戻り値）
        deactivate Optics
      Optics -> Util : get_projection
      note left : 既に書いているので省略
      Optics --> Telescope : proj(WCS)を返却
      deactivate Optics
    loop detectorsの要素数分
    Telescope -> Detector : get_footprint_as_polygon()
      activate Detector
      Detector -> Polygon : new
      note left : conerメソッドとかでいろいろやってるけど省略
      deactivate Detector
    end
    deactivate Telescope
  loop get_footprintsの戻り値の要素数分
  Telescope -> Telescope : axisにごにょごにょやってる
  end  
  deactivate Telescope
end  
Client -> pyplot : show()
end
group 7
loop 生成したTelescope分
Client -> Telescope : display_focal_plane
  activate Telescope
  Telescope -> Optics : get_fov_patch() 
  Telescope -> Source : convert_skycoord_to_sourcetable()
  Telescope -> Optics : imaging()
    activate Optics
    Optics -> tmp_SourceTable : new
    Optics -> tmp_sourceTable : apply_space_motion()
    Optics -> tmp_sourceTable : SkyCoordの取得
    tmp_sourceTable -> 
    deactivate Optics
  deactivate Telescope
end          
@enduml
```uml
