import os
import xml.etree.ElementTree as ET


images_path = './IU-XRay/images'
reports_path = './IU-XRay/reports'

try:
    os.makedirs(images_path)
    os.makedirs(reports_path)
except:
    print("path already exists")

# download PNG images
os.system("wget -P {}/ https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz".format(images_path))

# download reports
os.system("wget -P {}/ https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz".format(reports_path))

# unzip
os.system("tar -xzf {}/NLMCXR_png.tgz -C {}/".format(images_path, images_path))
os.system("tar -xzf {}/NLMCXR_reports.tgz -C {}/".format(reports_path, reports_path))
os.system("mv {}/ecgen-radiology/*.xml {}/".format(reports_path, reports_path))
os.system("rm -rf {}/ecgen-radiology".format(reports_path))

os.system("rm {}/NLMCXR_png.tgz".format(images_path))
os.system("rm {}/NLMCXR_reports.tgz".format(reports_path))

reports = os.listdir(reports_path)

reports.sort()

reports_with_no_image = []
reports_with_empty_sections = []
reports_with_no_impression = []
reports_with_no_findings = []

images_captions = {}
reports_with_images = {}
text_of_reports = {}

for report in reports:

    tree = ET.parse(os.path.join(reports_path, report))
    root = tree.getroot()
    img_ids = []
    # find the images of the report
    images = root.findall("parentImage")
    # if there aren't any ignore the report
    if len(images) == 0:
        reports_with_no_image.append(report)
    else:
        sections = root.find("MedlineCitation").find("Article").find("Abstract").findall("AbstractText")
        # find impression and findings sections
        for section in sections:
            if section.get("Label") == "FINDINGS":
                findings = section.text
            if section.get("Label") == "IMPRESSION":
                impression = section.text

        if impression is None and findings is None:
            reports_with_empty_sections.append(report)
        else:
            if impression is None:
                reports_with_no_impression.append(report)
                caption = findings
            elif findings is None:
                reports_with_no_findings.append(report)
                caption = impression
            else:
                caption = impression + " " + findings

            for image in images:
                images_captions[image.get("id") + ".png"] = caption
                img_ids.append(image.get("id") + ".png")

            reports_with_images[report] = img_ids
            text_of_reports[report] = caption

print("Found", len(reports_with_no_image), "reports with no associated image")
print("Found", len(reports_with_empty_sections), "reports with empty Impression and Findings sections")
print("Found", len(reports_with_no_impression), "reports with no Impression section")
print("Found", len(reports_with_no_findings), "reports with no Findings section")

print("Collected", len(images_captions), "image-caption pairs")
